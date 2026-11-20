"""Step 2-3: topology collection, validation and preprocessing.

Produces a deterministic directed-graph object for Abilene, GEANT (Geant2012)
and the canonical 14-node NSFNET. Every transformation is logged.
"""
import hashlib, json, os
import numpy as np
import networkx as nx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")

DEFAULT_CAP = 100.0   # Mbit/s (benchmark default, Phase 2 Step 3)
DEFAULT_PROP = 2.0    # ms one-way propagation delay

# Canonical NSFNET T1 backbone, 14 nodes and 21 links (as used in RL routing studies)
NSFNET_EDGES = [(0, 1), (0, 2), (0, 7), (1, 2), (1, 3), (2, 5), (3, 4), (3, 10), (4, 5),
                (4, 6), (5, 9), (5, 12), (6, 7), (7, 8), (8, 9), (8, 11), (8, 13),
                (10, 11), (10, 13), (11, 12), (12, 13)]

NETS = ["Abilene", "GEANT", "NSFNET"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


class Topology:
    """Directed graph with physical-link bookkeeping. A physical failure removes both directions."""

    def __init__(self, name, und_edges, n, log, caps=None, props=None, labels=None):
        self.name = name
        self.n = n
        self.labels = labels or [str(i) for i in range(n)]
        self.links = list(und_edges)                     # physical links (u<v)
        self.L = len(self.links)
        eu, ev, link_of = [], [], []
        for li, (u, v) in enumerate(self.links):
            eu += [u, v]; ev += [v, u]; link_of += [li, li]
        self.eu = np.array(eu); self.ev = np.array(ev)
        self.link_of = np.array(link_of)
        self.E = len(eu)
        self.cap = np.full(self.E, DEFAULT_CAP) if caps is None else np.repeat(np.asarray(caps, float), 2)
        self.prop = np.full(self.E, DEFAULT_PROP) if props is None else np.repeat(np.asarray(props, float), 2)
        self.eidx = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(self.eu, self.ev))}
        self.log = log
        g = nx.Graph(); g.add_nodes_from(range(n)); g.add_edges_from(self.links)
        self.ug = g
        self.bridges = set(tuple(sorted(b)) for b in nx.bridges(g))
        self.nonbridge_links = [i for i, l in enumerate(self.links) if tuple(sorted(l)) not in self.bridges]
        eb = nx.edge_betweenness_centrality(g)
        self.link_btw = np.array([eb.get(l, eb.get((l[1], l[0]), 0.0)) for l in self.links])
        self.deg = np.array([g.degree(i) for i in range(n)])

    def with_capacities(self, caps_per_link, tag):
        t = Topology(self.name + tag, self.links, self.n, self.log + [f"capacities replaced ({tag})"],
                     caps=caps_per_link, props=self.prop[::2], labels=self.labels)
        return t

    def summary(self):
        return dict(name=self.name, nodes=self.n, links=self.L, directed_edges=self.E,
                    bridges=len(self.bridges), nonbridge_links=len(self.nonbridge_links),
                    diameter=nx.diameter(self.ug), mean_degree=float(self.deg.mean()))


def _from_graphml(name, path):
    log = [f"read {os.path.basename(path)} sha256={sha256(path)[:16]}"]
    g = nx.read_graphml(path)
    nodes = sorted(g.nodes(), key=lambda s: int(s.lstrip("n")))     # deterministic mapping
    mp = {u: i for i, u in enumerate(nodes)}
    labels = [g.nodes[u].get("label", u) for u in nodes]
    log.append(f"raw nodes={g.number_of_nodes()} raw edge records={g.number_of_edges()}")
    mult = {}; selfloops = 0; dup = []
    for u, v in g.edges():
        if u == v:
            selfloops += 1; continue
        a, b = sorted((mp[u], mp[v]))
        if (a, b) in mult:
            dup.append((labels[a], labels[b]))
        mult[(a, b)] = mult.get((a, b), 0) + 1
    und = sorted(mult)
    caps = [DEFAULT_CAP * mult[e] for e in und]
    log.append(f"removed self-loops={selfloops}; parallel records aggregated with summed capacity={dup}")
    log.append("missing capacity/delay -> benchmark defaults 100 Mbit/s and 2 ms (labels kept only as metadata)")
    ug = nx.Graph(); ug.add_nodes_from(range(len(nodes))); ug.add_edges_from(und)
    assert nx.is_connected(ug), "topology must be connected"
    log.append("connectivity check passed")
    return Topology(name, und, len(nodes), log, caps=caps, labels=labels)


def load(name):
    if name == "Abilene":
        return _from_graphml("Abilene", os.path.join(RAW, "Abilene.graphml"))
    if name == "GEANT":
        return _from_graphml("GEANT", os.path.join(RAW, "Geant2012.graphml"))
    if name == "NSFNET":
        log = ["canonical 14-node NSFNET edge list embedded in code", "defaults 100 Mbit/s and 2 ms"]
        return Topology("NSFNET", sorted(tuple(sorted(e)) for e in NSFNET_EDGES), 14, log)
    raise ValueError(name)


def build_manifest():
    os.makedirs(PROC, exist_ok=True)
    out = {}
    for n in NETS:
        t = load(n)
        s = t.summary(); s["log"] = t.log
        out[n] = s
    with open(os.path.join(PROC, "topology_manifest.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print(json.dumps(build_manifest(), indent=1))
