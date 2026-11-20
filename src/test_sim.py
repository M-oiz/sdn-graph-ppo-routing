"""Stage 2 checks on a hand-verifiable graph (Phase 2 Section 6.1)."""
import numpy as np
from topo import Topology
from netsim import Env, TICK, recovery_metrics
from methods import StaticSP, LoadAware


def square():
    # 0-1-2 and 0-3-2 : two disjoint 2-hop routes, plus 1-3 diagonal
    return Topology("TestSquare", [(0, 1), (1, 2), (0, 3), (2, 3)], 4, ["hand-made"])


def wl(ts, src, dst, rate, dur, cls=None):
    n = len(ts)
    return dict(t=np.array(ts, float), src=np.array(src), dst=np.array(dst), rate=np.array(rate, float),
                dur=np.array(dur, float), cls=np.zeros(n, int) if cls is None else np.array(cls),
                rate_seed=1, lam=0, mcf=0)


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name)
    assert cond


t = square()
# 1) reservation bookkeeping and release
w = wl([0.1, 0.2], [0, 0], [2, 2], [60, 60], [5, 5])
e = Env(t, w, StaticSP(), ep_len=10)
e.run()
check("both flows admitted (second via capacity-aware fallback)", e.n_reject == 0 and e.n_fallback == 1)
check("reservations fully released after flows end", np.allclose(e.resv, 0))
# 2) admission rejection when no capacity left anywhere
w = wl([0.1, 0.2, 0.3], [0, 0, 0], [2, 2, 2], [60, 60, 60], [5, 5, 5])
e = Env(t, w, StaticSP(), ep_len=10); e.run()
check("third 60 Mbit/s flow rejected (only 2x100 Mbit/s cut)", e.n_reject == 1)
# 3) failure removes both directions and reroutes affected flow
w = wl([0.1], [0], [2], [10], [30])
e = Env(t, w, StaticSP(), ep_len=20, fail_links=[0], fail_start=5.0, fail_dur=50.0)
e.run()
check("failure removed both directed edges of the link", (~e.avail[:t.E][t.link_of == 0]).sum() == 2 and e.avail[:t.E].sum() == t.E - 2)
check("affected flow rerouted once", e.n_reroute_fail == 1 and e.n_dropped == 0)
lat = e.restore_lat[0]
check("restoration latency = detection + decision + install (100-110 ms)", 0.1 < lat < 0.11)
# delivered = sent minus outage share of one tick
lost = e.sent[0] - e.deliv[0]
expect = 10 * e.mult[0] * 0 + lost
check("loss equals outage only (no congestion): %.3f Mbit" % lost, 0.5 < lost < 1.5)
# 4) congestion: two flows reserved 50 each on one 100 link can overflow with rate noise
w = wl([20.1, 20.1], [0, 0], [1, 1], [50, 50], [10, 10])
e = Env(t, w, StaticSP(), ep_len=32); r = e.run()
check("queueing delay is positive under high utilisation", r["delay_mean"] > 2.0)
# 5) load-aware heuristic splits two flows over disjoint routes
w = wl([0.1, 0.2], [0, 0], [2, 2], [40, 40], [5, 5])
e = Env(t, w, LoadAware(), ep_len=8); e.run()
check("load-aware uses both disjoint paths", e.n_fallback == 0 and len({e.pnodes[0], e.pnodes[1]}) == 2)
# 6) recovery metric on synthetic series
ref = np.full(480, 100.0); obs = ref.copy(); obs[240:244] = 50
rec, defi, lvl = recovery_metrics(obs, ref)
check("recovery time 1.0 s and deficit 50 Mbit on synthetic dip", abs(rec - 1.0) < 1e-9 and abs(defi - 50) < 1e-9)
print("all checks passed")
