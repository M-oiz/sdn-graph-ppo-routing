"""Step 2: seeded synthetic workloads (uniform, hotspot, bursty) and load calibration.

Offered load is a fraction of the intact topology's maximum concurrent flow (MCF)
for the same demand proportions, solved as a source-aggregated multicommodity LP.
"""
import functools
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import lil_matrix

EP_LEN = 120.0
RATE_LO, RATE_HI = 1.0, 5.0          # Mbit/s
DUR_MEAN, DUR_LO, DUR_HI = 10.0, 5.0, 20.0
REGIMES = ["uniform", "hotspot", "bursty"]
NET_ID = {"Abilene": 1, "GEANT": 2, "NSFNET": 3}


def trunc_exp_mean():
    a, b, m = DUR_LO, DUR_HI, DUR_MEAN
    # mean of exponential(mean m) truncated to [a,b]
    za, zb = np.exp(-a / m), np.exp(-b / m)
    return ((a + m) * za - (b + m) * zb) / (za - zb)


def sample_dur(rng, size):
    m = DUR_MEAN
    u = rng.random(size)
    za, zb = np.exp(-DUR_LO / m), np.exp(-DUR_HI / m)
    return -m * np.log(za - u * (za - zb))


def demand_matrix(n, regime, rng):
    d = np.ones((n, n)); np.fill_diagonal(d, 0)
    hot = None
    if regime == "hotspot":
        k = max(1, int(round(0.2 * n)))
        hot = rng.choice(n, k, replace=False)
        # 40% of requests target the hot subset, 60% spread over all destinations
        p_hot = np.zeros(n); p_hot[hot] = 1
        p_all = np.ones(n)
        d = np.zeros((n, n))
        for s in range(n):
            ph = p_hot.copy(); ph[s] = 0
            pa = p_all.copy(); pa[s] = 0
            row = 0.6 * pa / pa.sum()
            if ph.sum() > 0:
                row += 0.4 * ph / ph.sum()
            else:
                row += 0.4 * pa / pa.sum()
            d[s] = row
    return d / d.sum(), hot


def max_concurrent_flow(topo, dem, fail_links=()):
    """Max lambda s.t. lambda*dem is routable (fractional, source-aggregated)."""
    n, E = topo.n, topo.E
    alive = np.ones(E, bool)
    for l in fail_links:
        alive[topo.link_of == l] = False
    edges = np.where(alive)[0]
    m = len(edges)
    nv = n * m + 1                       # f[s,e] ... , lambda
    A = lil_matrix((n * n, nv)); b = np.zeros(n * n)
    for s in range(n):
        for j, e in enumerate(edges):
            u, v = topo.eu[e], topo.ev[e]
            col = s * m + j
            A[s * n + v, col] += 1        # inflow to v
            A[s * n + u, col] -= 1        # outflow from u
        for v in range(n):
            if v == s:
                A[s * n + v, nv - 1] = dem[s].sum()     # source emits total
                # inflow-outflow at source = -sum demand -> move lambda term
            else:
                A[s * n + v, nv - 1] = -dem[s, v]
    # constraint: inflow-outflow - lambda*dem = 0 at sinks; at source: inflow-outflow + lambda*sum = 0
    Aub = lil_matrix((m, nv)); bub = np.zeros(m)
    for j, e in enumerate(edges):
        for s in range(n):
            Aub[j, s * m + j] = 1
        bub[j] = topo.cap[e]
    c = np.zeros(nv); c[-1] = -1
    res = linprog(c, A_ub=Aub.tocsr(), b_ub=bub, A_eq=A.tocsr(), b_eq=b, bounds=(0, None), method="highs")
    assert res.status == 0, res.message
    return -res.fun                       # total Mbit/s of concurrent demand


@functools.lru_cache(maxsize=None)
def _mcf_cached(name, regime, hot_key, caps_key):
    from topo import load
    topo = load(name.split("#")[0])
    if caps_key is not None:
        topo = topo.with_capacities(np.array(caps_key), "#het")
    rng = np.random.default_rng(0)
    if regime == "hotspot":
        n = topo.n
        hot = np.array(hot_key)
        d = np.zeros((n, n))
        for s in range(n):
            ph = np.zeros(n); ph[hot] = 1; ph[s] = 0
            pa = np.ones(n); pa[s] = 0
            row = 0.6 * pa / pa.sum() + (0.4 * ph / ph.sum() if ph.sum() > 0 else 0.4 * pa / pa.sum())
            d[s] = row
        d /= d.sum()
    else:
        d, _ = demand_matrix(topo.n, "uniform", rng)
    return max_concurrent_flow(topo, d)


def make_workload(topo, seed, load, regime, burst=False, ep_len=EP_LEN):
    """Return arrays of flows sorted by arrival: t, src, dst, rate, dur, cls, plus meta."""
    rng = np.random.default_rng([NET_ID[topo.name.split("#")[0]], seed, int(load * 100), REGIMES.index(regime)])
    d, hot = demand_matrix(topo.n, regime, rng)
    caps_key = None
    if "#het" in topo.name:
        caps_key = tuple(np.round(topo.cap[::2], 3))
    T = _mcf_cached(topo.name, regime, tuple(sorted(hot.tolist())) if hot is not None else None, caps_key)
    mean_rate = (RATE_LO + RATE_HI) / 2
    lam = load * T / (mean_rate * trunc_exp_mean())     # arrivals per second
    burst = burst or regime == "bursty"
    # thinning for time-varying arrival rate
    lam_max = lam * (2 if burst else 1)
    nmax = rng.poisson(lam_max * ep_len)
    t = np.sort(rng.random(nmax) * ep_len)
    if burst:
        keep_p = np.where((t >= 50) & (t < 70), 1.0, 0.5)
        t = t[rng.random(len(t)) < keep_p]
    N = len(t)
    pairs = rng.choice(topo.n * topo.n, N, p=d.ravel())
    src, dst = pairs // topo.n, pairs % topo.n
    rate = rng.uniform(RATE_LO, RATE_HI, N)
    dur = sample_dur(rng, N)
    cls = (rng.random(N) < 0.5).astype(int)            # 0 interactive, 1 bulk
    rate_seed = int(rng.integers(1 << 31))
    return dict(t=t, src=src, dst=dst, rate=rate, dur=dur, cls=cls, lam=lam, mcf=T,
                regime=regime, burst=burst, rate_seed=rate_seed, hot=None if hot is None else hot.tolist())


if __name__ == "__main__":
    from topo import load, NETS
    print("E[dur]", trunc_exp_mean())
    for n in NETS:
        t = load(n)
        for r in REGIMES:
            w = make_workload(t, 801, 0.95, r)
            print(n, r, "MCF=%.0f Mbit/s" % w["mcf"], "lam=%.1f/s" % w["lam"], "flows", len(w["t"]))
