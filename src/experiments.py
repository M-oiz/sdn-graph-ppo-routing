"""Steps 7, 10 and 11: frozen-policy evaluation matrix, ablations and stress tests.

Usage: python3 experiments.py main|ablation|stress
Rows are appended to results/<suite>.jsonl; finished jobs are skipped on restart.
"""
import os, sys, json, time, hashlib
from multiprocessing import Pool
import numpy as np
import networkx as nx
from topo import load, NETS
from workload import make_workload, REGIMES
from netsim import Env, PathCache, recovery_metrics
from methods import StaticSP, ECMP, LoadAware
from agents import Learner, LearnedMethod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
MODELS = os.path.join(RES, "models")
LOADS = [0.4, 0.7, 0.95]
CONDS = ["none", "random", "targeted", "double"]
TEST_W = [800, 801, 802, 803, 804]
NET_ID = {"Abilene": 1, "GEANT": 2, "NSFNET": 3}

_topo, _pc, _models = {}, {}, {}


def topo_of(net, het=False):
    key = (net, het)
    if key not in _topo:
        t = load(net)
        if het:
            rng = np.random.default_rng([NET_ID[net], 77])
            t = t.with_capacities(rng.uniform(50, 200, t.L).round(1), "#het")
        _topo[key] = t; _pc[key] = PathCache(t)
    return _topo[key], _pc[key]


def regime_of(wseed):
    return REGIMES[(wseed - 800) % 3]


def cond_links(t, cond, wseed):
    rng = np.random.default_rng([NET_ID[t.name.split("#")[0]], wseed, 5])
    nb = t.nonbridge_links
    if cond == "none":
        return []
    if cond == "random":
        return [int(rng.choice(nb))]
    if cond == "targeted":
        return [int(nb[int(np.argmax(t.link_btw[nb]))])]
    if cond == "double":
        for _ in range(500):
            a, b = rng.choice(nb, 2, replace=False)
            g = t.ug.copy(); g.remove_edges_from([t.links[a], t.links[b]])
            if nx.is_connected(g):
                return sorted([int(a), int(b)])
    raise ValueError(cond)


def make_method(spec, t):
    m = spec["method"]
    if m == "B1":
        return StaticSP()
    if m == "B2":
        return ECMP()
    if m == "B3":
        return LoadAware()
    path = os.path.join(MODELS, f"{spec['tag']}_{spec['src']}_s{spec['mseed']}.pkl")
    if path not in _models:
        _models[path] = Learner.load(path)
    return LearnedMethod(_models[path]).bind(t)


def run_job(spec):
    het = spec.get("het", False)
    t, pc = topo_of(spec["net"], het)
    reg = spec.get("regime") or regime_of(spec["wseed"])
    w = make_workload(t, spec["wseed"], spec["load"], reg, burst=spec.get("burst", False))
    fl = cond_links(t, spec["cond"], spec["wseed"])
    M = make_method(spec, t)
    K = spec.get("K", 4)
    kw = dict(K=K, hold_down=spec.get("hold", True), fallback=spec.get("fallback", True),
              telem_delay=spec.get("tdelay", 0.0), miss_prob=spec.get("miss", 0.0), pcache=pc)
    e = Env(t, w, M, fail_links=fl, **kw)
    r = e.run()
    row = dict(spec); row.update(r); row["regime"] = reg; row["fail_links"] = fl
    if spec.get("ref") and fl:
        ref = Env(t, w, M, fail_links=fl, warm_failed=True, **kw)
        ref.run()
        rec, deficit, lvl = recovery_metrics(e.tick_goodput, ref.tick_goodput)
        row.update(recovery_s=rec, deficit_mbit=deficit, ref_level=lvl, censored=bool(np.isnan(rec)))
        if spec.get("keep_series"):
            row["gp_series"] = np.round(e.tick_goodput, 2).tolist()
            row["gp_ref"] = np.round(ref.tick_goodput, 2).tolist()
    return row


def key_of(spec):
    return hashlib.md5(json.dumps(spec, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ job lists
def main_jobs():
    J = []
    for net in NETS:
        for m in ["B1", "B2", "B3"]:
            for ld in LOADS:
                for c in CONDS:
                    for ws in TEST_W:
                        J.append(dict(suite="main", fold=None, src=None, method=m, mseed=0, net=net, role="all",
                                      load=ld, cond=c, wseed=ws, ref=c != "none", keep_series=c == "targeted"))
    for src in NETS:
        for tag, m in [("proposed", "GPPO"), ("mlp", "B4")]:
            for ms in range(3):
                for net in NETS:
                    for ld in LOADS:
                        for c in CONDS:
                            for ws in TEST_W:
                                J.append(dict(suite="main", fold=src, src=src, tag=tag, method=m, mseed=ms, net=net,
                                              role="source" if net == src else "target", load=ld, cond=c, wseed=ws,
                                              ref=(c != "none" and ms == 0), keep_series=(c == "targeted" and ms == 0)))
    return J


ABL = [("nofail", "proposed-no-failure-training", {}), ("nochurn", "no-churn-penalty", {}),
       ("K2", "K=2", {"K": 2}), ("K8", "K=8", {"K": 8}),
       ("proposed", "no-hold-down", {"hold": False}), ("proposed", "no-fallback", {"fallback": False})]


ABL_FOLDS = ["Abilene", "NSFNET"]


def ablation_jobs():
    J = []
    for src in ABL_FOLDS:
        for tag, var, extra in ABL:
            for ms in range(2):
                for net in NETS:
                    if net == src:
                        continue
                    for ld in [0.7, 0.95]:
                        for c in ["none", "targeted", "double"]:
                            for ws in TEST_W:
                                d = dict(suite="ablation", fold=src, src=src, tag=tag, method="GPPO", variant=var,
                                         mseed=ms, net=net, role="target", load=ld, cond=c, wseed=ws,
                                         ref=(tag == "nofail" and ms == 0 and c != "none"))
                                d.update(extra)
                                J.append(d)
    return J


STRESS = [("telemetry 0 ms", {}), ("telemetry +250 ms", {"tdelay": 0.25}), ("telemetry +1000 ms", {"tdelay": 1.0}),
          ("5% missing polls", {"miss": 0.05}), ("heterogeneous 50-200 Mbit/s", {"het": True}),
          ("burst + single failure", {"burst": True})]


def stress_jobs():
    J = []
    W = [800, 801]
    for sname, extra in STRESS:
        for net in NETS:
            for m in ["B1", "B2", "B3"]:
                for ws in W:
                    d = dict(suite="stress", stress=sname, fold=None, src=None, method=m, mseed=0, net=net, role="all",
                             load=0.95, cond="targeted", wseed=ws, ref=True)
                    d.update(extra); J.append(d)
        for src in NETS:
            for tag, m in [("proposed", "GPPO"), ("mlp", "B4")]:
                for ms in range(3):
                    for net in NETS:
                        if net == src:
                            continue
                        for ws in W:
                            d = dict(suite="stress", stress=sname, fold=src, src=src, tag=tag, method=m, mseed=ms, net=net,
                                     role="target", load=0.95, cond="targeted", wseed=ws, ref=True)
                            d.update(extra); J.append(d)
    return J


def run_suite(name, jobs, procs=2):
    out = os.path.join(RES, f"{name}.jsonl")
    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                done.add(json.loads(line)["_key"])
    todo = [j for j in jobs if key_of(j) not in done]
    # heavy (GEANT) jobs first for better load balance
    todo.sort(key=lambda j: (j["net"] != "GEANT", j.get("src") or "", j.get("tag") or "", j.get("mseed", 0)))
    print(f"{name}: {len(jobs)} jobs, {len(todo)} to run", flush=True)
    t0 = time.time()
    with Pool(procs) as p, open(out, "a") as f:
        for i, row in enumerate(p.imap_unordered(_wrap, todo, chunksize=4)):
            f.write(json.dumps(row) + "\n"); f.flush()
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(todo)}  {time.time()-t0:.0f}s", flush=True)


def _wrap(spec):
    row = run_job(spec)
    row["_key"] = key_of(spec)
    for k, v in list(row.items()):
        if isinstance(v, (np.floating, np.integer)):
            row[k] = v.item()
    return row


if __name__ == "__main__":
    suite = sys.argv[1]
    jobs = dict(main=main_jobs, ablation=ablation_jobs, stress=stress_jobs)[suite]()
    run_suite(suite, jobs)
