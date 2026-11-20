"""Step 6: controlled baselines B1-B3 (B4 and the proposed policy live in agents.py)."""
import numpy as np


def k_with_prev(env, f, s, d, K, kind):
    c = list(env.pc.ksp(env.failed, s, d, K))
    if kind == "reroute" and env.pnodes[f] is not None:
        prev = env.pnodes[f]
        if all(p[0] != prev for p in c):
            pe = env.path[f][env.path[f] < env.t.E]
            if len(c) >= K:
                c[-1] = (prev, pe)
            else:
                c.append((prev, pe))
    return c


class StaticSP:
    """B1: Dijkstra on fixed inverse-capacity weights; recomputed after a detected failure."""
    name = "B1 Static SP"; reroutes = False

    def candidates(self, env, f, s, d, kind):
        p = env.pc.sp(env.failed, s, d)
        return [p] if p is not None else []

    def choose(self, env, f, cands, mask, resid, kind):
        return 0 if mask[0] else None


class ECMP:
    """B2: flow pinned to a hashed equal-cost shortest path."""
    name = "B2 ECMP"; reroutes = False

    def candidates(self, env, f, s, d, kind):
        ps = env.pc.ecmp(env.failed, s, d)
        if not ps:
            return []
        h = (int(f) * 2654435761) % (2 ** 32)
        return [ps[h % len(ps)]]

    def choose(self, env, f, cands, mask, resid, kind):
        return 0 if mask[0] else None


class LoadAware:
    """B3: least-utilised feasible path among the same K candidates (tie: propagation delay)."""
    name = "B3 Load-aware"; reroutes = True

    def __init__(self, K=4):
        self.K = K

    def candidates(self, env, f, s, d, kind):
        return k_with_prev(env, f, s, d, env.K, kind)

    def choose(self, env, f, cands, mask, resid, kind):
        best, bk = None, None
        rate = env.w["rate"][f]
        for i, (nodes, e) in enumerate(cands):
            if not mask[i]:
                continue
            u = float((1 - (resid[e] - rate) / env.cap[e]).max())
            key = (round(u, 9), float(env.prop[e].sum()), i)
            if bk is None or key < bk:
                best, bk = i, key
        return best
