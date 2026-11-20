"""Flow-level SDN routing simulator (Steps 5-8 of the Phase 2 methodology).

Time advances in 0.25 s ticks; routing epochs are 1 s; telemetry is polled every tick
(250 ms). Bandwidth is reserved per admitted flow, but each flow's real sending rate
fluctuates every epoch, so links that are reserved close to capacity can overflow and
create queueing delay and loss. All methods share the same workload, failure schedule,
telemetry, admission check, fallback and controller timing model.
"""
import heapq, time
import numpy as np
import networkx as nx

TICK = 0.25
EPOCH = 1.0
TPE = int(round(EPOCH / TICK))
WARMUP = 20.0
LMAX = 24
DELAY_TARGET = np.array([100.0, 500.0])     # ms, p95 one-way: interactive, bulk
LOSS_TARGET = np.array([0.01, 0.02])
Q_SCALE = 1.0                               # ms; queueing ~ Q_SCALE * rho/(1-rho)
BUF_MS = 50.0                               # full-buffer queueing delay when overloaded
DETECT = 0.10                               # s, failure detection delay (same for all)
INSTALL = 0.005                             # s, rule install round trip per decision
FLOWMOD_B, BARRIER_B = 144, 16              # bytes per OpenFlow message
RATE_SHAPE = 16.0                           # gamma shape for per-epoch rate fluctuation (cv 0.25)
REWARD_W = dict(S=0.40, D=0.20, L=0.15, U=0.10, C=0.15)


class PathCache:
    """Yen K-shortest loop-free paths (propagation-delay cost) per failure set."""

    def __init__(self, topo):
        self.t = topo
        self.g = {}
        self.cache = {}

    def graph(self, failed):
        if failed not in self.g:
            g = nx.DiGraph()
            g.add_nodes_from(range(self.t.n))
            for e in range(self.t.E):
                if self.t.link_of[e] in failed:
                    continue
                g.add_edge(int(self.t.eu[e]), int(self.t.ev[e]), w=float(self.t.prop[e]),
                           ic=1000.0 / float(self.t.cap[e]), e=e)
            self.g[failed] = g
        return self.g[failed]

    def to_edges(self, nodes):
        return np.array([self.t.eidx[(nodes[i], nodes[i + 1])] for i in range(len(nodes) - 1)], dtype=np.int64)

    def ksp(self, failed, s, d, K):
        key = ("k", failed, s, d, K)
        if key not in self.cache:
            g = self.graph(failed)
            out = []
            try:
                for p in nx.shortest_simple_paths(g, s, d, weight="w"):
                    out.append((tuple(p), self.to_edges(p)))
                    if len(out) >= K:
                        break
            except nx.NetworkXNoPath:
                pass
            self.cache[key] = out
        return self.cache[key]

    def sp(self, failed, s, d):
        key = ("sp", failed, s, d)
        if key not in self.cache:
            g = self.graph(failed)
            try:
                p = nx.shortest_path(g, s, d, weight="ic")
                self.cache[key] = (tuple(p), self.to_edges(p))
            except nx.NetworkXNoPath:
                self.cache[key] = None
        return self.cache[key]

    def ecmp(self, failed, s, d):
        key = ("ecmp", failed, s, d)
        if key not in self.cache:
            g = self.graph(failed)
            try:
                ps = sorted(nx.all_shortest_paths(g, s, d, weight="ic"))
                self.cache[key] = [(tuple(p), self.to_edges(p)) for p in ps]
            except nx.NetworkXNoPath:
                self.cache[key] = []
        return self.cache[key]


class Env:
    def __init__(self, topo, wl, method, K=4, fail_links=(), fail_start=60.0, fail_dur=30.0,
                 warm_failed=False, hold_down=True, fallback=True, telem_delay=0.0, miss_prob=0.0,
                 ep_len=120.0, learner=None, reward_w=None, reward_mode="final", max_reroute=20,
                 hot_thr=0.9, seed=0, pcache=None, record_series=True):
        self.t = topo; self.w = wl; self.m = method
        self.K = K; self.hold = hold_down; self.fallback = fallback
        self.fail_links = tuple(sorted(fail_links)); self.fail_start = fail_start; self.fail_dur = fail_dur
        self.warm_failed = warm_failed
        self.telem_delay = telem_delay; self.miss_prob = miss_prob
        self.ep_len = ep_len; self.learner = learner
        self.rw = dict(REWARD_W if reward_w is None else reward_w); self.reward_mode = reward_mode
        self.max_reroute = max_reroute; self.hot_thr = hot_thr
        self.rng = np.random.default_rng([seed, wl["rate_seed"]])
        self.pc = pcache or PathCache(topo)
        self.record_series = record_series
        E = topo.E
        self.failed = frozenset(self.fail_links) if warm_failed else frozenset()
        self.avail = np.ones(E + 1, bool)
        if warm_failed:
            for l in self.fail_links:
                self.avail[:E][topo.link_of == l] = False
        self.cap = np.append(topo.cap, 1e9)
        self.prop = np.append(topo.prop, 0.0)
        self.resv = np.zeros(E + 1)
        N = len(wl["t"])
        self.N = N
        self.path = np.full((N, LMAX), E, dtype=np.int64)
        self.pnodes = [None] * N
        self.status = np.zeros(N, np.int8)      # 0 future, 1 active, 2 done, 3 rejected, 4 dropped
        self.inst_t = np.zeros(N)
        self.out_until = np.zeros(N)
        self.mult = np.ones(N)
        # per-flow, per-epoch sending-rate multipliers drawn once from the workload seed so that
        # every method sees exactly the same traffic (paired comparison)
        self.G = np.random.default_rng(wl["rate_seed"]).gamma(RATE_SHAPE, 1 / RATE_SHAPE, (N, int(ep_len / EPOCH) + 2))
        self.sent = np.zeros(N); self.deliv = np.zeros(N)
        self.dl_sum = np.zeros(N); self.dl_cnt = np.zeros(N)
        self.dsamp = [[] for _ in range(N)]
        self.ep_sent = np.zeros(N); self.ep_lost = np.zeros(N); self.ep_dmax = np.zeros(N); self.ep_dsum = np.zeros(N); self.ep_cnt = np.zeros(N)
        self.end = wl["t"] + wl["dur"]
        self.active = []
        self.next_arr = 0
        # telemetry
        self.tel_hist = [(-1e9, np.zeros(E + 1), np.zeros(E + 1))]
        self.tel_now = self.tel_hist[0]
        # stats
        self.dec_times = []          # (kind, cand_s, infer_s, check_s, total_s)
        self.flowmods = 0; self.barriers = 0
        self.n_fallback = 0; self.n_masked = 0; self.n_decisions = 0; self.n_reject = 0; self.n_dropped = 0
        self.n_reroute_disc = 0; self.n_reroute_fail = 0; self.n_learner_invalid = 0
        self.restore_lat = []
        self.hist = np.zeros(2001)   # delivered-Mbit-weighted delay histogram, 0.5 ms bins
        self.series = dict(t=[], goodput=[], maxutil=[], churn=[], active=[], loss=[])
        self.tick_goodput = []
        self.ep_disc = 0; self.ep_rejects = 0; self.util_acc = np.zeros(E + 1); self.ep_ticks = 0
        self.ctrl_time = 0.0
        self.fw_sent = 0.0; self.fw_lost = 0.0
        self.now = 0.0

    # ------------------------------------------------------------------ helpers
    def telemetry(self):
        return self.tel_now

    def residual(self, exclude=None):
        r = self.cap - self.resv
        if exclude is not None:
            pe = self.path[exclude]
            np.add.at(r, pe[pe < self.t.E], self.w["rate"][exclude])
        return r

    def feasible(self, edges, rate, resid):
        return bool(np.all(self.avail[edges]) and np.all(resid[edges] >= rate - 1e-9))

    def fallback_path(self, s, d, rate, resid):
        """Capacity-aware Dijkstra over the full available graph (prop-delay cost)."""
        t = self.t
        dist = {s: 0.0}; prev = {}
        pq = [(0.0, s)]; seen = set()
        adj = self._adj()
        while pq:
            du, u = heapq.heappop(pq)
            if u in seen:
                continue
            seen.add(u)
            if u == d:
                break
            for v, e in adj[u]:
                if not self.avail[e] or resid[e] < rate - 1e-9:
                    continue
                nd = du + t.prop[e]
                if nd < dist.get(v, 1e18):
                    dist[v] = nd; prev[v] = (u, e); heapq.heappush(pq, (nd, v))
        if d not in seen:
            return None
        nodes = [d]; edges = []
        while nodes[-1] != s:
            u, e = prev[nodes[-1]]; edges.append(e); nodes.append(u)
        return tuple(reversed(nodes)), np.array(list(reversed(edges)), dtype=np.int64)

    def _adj(self):
        if not hasattr(self, "_adjl"):
            a = [[] for _ in range(self.t.n)]
            for e in range(self.t.E):
                a[self.t.eu[e]].append((int(self.t.ev[e]), e))
            self._adjl = a
        return self._adjl

    def install(self, f, nodes, edges, kind):
        E = self.t.E
        old = self.path[f]
        if self.status[f] == 1:
            ok = old[old < E]
            np.subtract.at(self.resv, ok, self.w["rate"][f])
            self.flowmods += len(self.pnodes[f]) - 1          # delete old rules
        self.path[f] = E
        self.path[f, :len(edges)] = edges
        np.add.at(self.resv, edges, self.w["rate"][f])
        self.pnodes[f] = nodes
        self.flowmods += len(nodes) - 1
        self.barriers += 2 * (len(nodes) - 1)
        self.inst_t[f] = self.now

    def release(self, f):
        pe = self.path[f]
        np.subtract.at(self.resv, pe[pe < self.t.E], self.w["rate"][f])
        self.flowmods += len(self.pnodes[f]) - 1 if self.pnodes[f] else 0

    def path_cost(self, edges, resid, rate):
        u = 1 - (resid[edges] - rate) / self.cap[edges]
        return float(u.max() + 0.01 * len(edges))

    # ------------------------------------------------------------------ decision
    def decide(self, f, kind):
        """kind: 'arrival' | 'failure' | 'reroute'. Returns True if the flow has a path."""
        t0 = time.perf_counter()
        s, d, rate = int(self.w["src"][f]), int(self.w["dst"][f]), self.w["rate"][f]
        resid = self.residual(exclude=f if kind != "arrival" else None)
        cands = self.m.candidates(self, f, s, d, kind)
        t1 = time.perf_counter()
        mask = np.array([self.feasible(e, rate, resid) for _, e in cands], bool)
        self.n_decisions += 1
        if len(cands) and not mask.all():
            self.n_masked += 1
        choice = None
        t2 = time.perf_counter()
        if mask.any():
            choice = self.m.choose(self, f, cands, mask, resid, kind)
        t3 = time.perf_counter()
        if kind == "reroute":
            prev = self.pnodes[f]
            if choice is None or cands[choice][0] == prev:
                self._dec_time(kind, t0, t1, t2, t3)
                return True
            if self.hold:
                if self.path_cost(cands[choice][1], resid, rate) > 0.95 * self.path_cost(self.path[f][self.path[f] < self.t.E], resid, rate):
                    self._dec_time(kind, t0, t1, t2, t3)
                    return True
            self.install(f, cands[choice][0], cands[choice][1], kind)
            self.n_reroute_disc += 1; self.ep_disc += 1
            self._dec_time(kind, t0, t1, t2, t3)
            return True
        if choice is None and self.fallback:
            fp = self.fallback_path(s, d, rate, resid)
            if fp is not None:
                self.n_fallback += 1
                nodes, edges = fp
                self.install(f, nodes, edges, kind)
                self._dec_time(kind, t0, t1, t2, time.perf_counter())
                return True
        if choice is None:
            self._dec_time(kind, t0, t1, t2, time.perf_counter())
            return False
        self.install(f, cands[choice][0], cands[choice][1], kind)
        self._dec_time(kind, t0, t1, t2, t3)
        return True

    def _dec_time(self, kind, t0, t1, t2, t3):
        tot = time.perf_counter() - t0
        self.ctrl_time += tot
        self.dec_times.append((kind, t1 - t0, t3 - t2, (t2 - t1), tot))
        self._last_dec = tot

    # ------------------------------------------------------------------ main loop
    def run(self):
        t = self.t; E = t.E; w = self.w
        nticks = int(round(self.ep_len / TICK))
        fail_on = (not self.warm_failed) and len(self.fail_links) > 0
        for k in range(nticks):
            t0 = k * TICK; t1 = t0 + TICK; self.now = t0
            # --- failure events
            if fail_on and abs(t0 - self.fail_start) < 1e-9:
                self._fail_event()
            if fail_on and abs(t0 - (self.fail_start + self.fail_dur)) < 1e-9:
                self.failed = frozenset(); self.avail[:] = True
            # --- departures
            still = []
            for f in self.active:
                if self.end[f] <= t0:
                    self.status[f] = 2; self.release(f)
                else:
                    still.append(f)
            self.active = still
            # --- arrivals in this tick
            while self.next_arr < self.N and w["t"][self.next_arr] < t1:
                f = self.next_arr; self.next_arr += 1
                self.now = w["t"][f]
                if self.decide(f, "arrival"):
                    self.status[f] = 1; self.active.append(f)
                    self.mult[f] = self.G[f, int(t0 // EPOCH)]
                else:
                    self.status[f] = 3; self.n_reject += 1; self.ep_rejects += 1
            self.now = t0
            # --- data plane for this tick
            self._tick(t0, t1)
            # --- telemetry poll
            if self.rng.random() >= self.miss_prob:
                self.tel_hist.append((t1, self._util.copy(), self._eloss.copy()))
            avail_t = [h for h in self.tel_hist if h[0] <= t1 - self.telem_delay + 1e-9]
            self.tel_now = avail_t[-1] if avail_t else self.tel_hist[0]
            if len(self.tel_hist) > 12:
                self.tel_hist = self.tel_hist[-12:]
            # --- epoch end
            if (k + 1) % TPE == 0:
                self.now = t1
                self._epoch_end(t1)
        return self.finish()

    def _fail_event(self):
        t = self.t; E = t.E
        self.failed = frozenset(self.fail_links)
        for l in self.fail_links:
            self.avail[:E][t.link_of == l] = False
        affected = [f for f in self.active if not np.all(self.avail[self.path[f][self.path[f] < E]])]
        clock = self.fail_start + DETECT
        keep = []
        aff = set(affected)
        for f in affected:
            ok = self.decide(f, "failure")
            clock += self._last_dec + INSTALL
            if ok:
                self.out_until[f] = clock
                self.restore_lat.append(clock - self.fail_start)
                self.n_reroute_fail += 1
            else:
                self.status[f] = 4; self.n_dropped += 1; self.release(f)
                self.out_until[f] = self.ep_len + 1
                self.restore_lat.append(np.nan)
        self.active = [f for f in self.active if self.status[f] == 1]

    def _tick(self, t0, t1):
        t = self.t; E = t.E; w = self.w
        A = np.array(self.active, dtype=np.int64)
        if len(A) == 0:
            self._util = np.zeros(E + 1); self._eloss = np.zeros(E + 1)
            self.tick_goodput.append(0.0); self.util_acc += 0; self.ep_ticks += 1
            return
        frac = (np.minimum(self.end[A], t1) - np.maximum(w["t"][A], t0)) / TICK
        frac = np.clip(frac, 0, 1)
        r = w["rate"][A] * self.mult[A] * frac                  # Mbit/s averaged over tick
        pe = self.path[A]
        load = np.bincount(pe.ravel(), weights=np.repeat(r, LMAX), minlength=E + 1)
        load[E] = 0
        rho = load / self.cap
        eloss = np.where(rho > 1, 1 - 1 / np.maximum(rho, 1e-9), 0.0)
        rc = np.minimum(rho, 0.98)
        q = np.where(rho >= 1, BUF_MS, Q_SCALE * rc / (1 - rc))
        q[E] = 0
        edelay = self.prop + q
        pdelay = edelay[pe].sum(1)
        ploss = 1 - np.prod(1 - eloss[pe], axis=1)
        alive = 1 - np.clip((self.out_until[A] - t0) / TICK, 0, 1)
        sent = r * TICK
        dl = sent * (1 - ploss) * alive
        self.sent[A] += sent; self.deliv[A] += dl
        if self.fail_links and self.fail_start <= t0 < self.fail_start + self.fail_dur:
            self.fw_sent += sent.sum(); self.fw_lost += (sent - dl).sum()
        has = (dl > 0)
        for i in np.where(has)[0]:
            self.dsamp[A[i]].append(pdelay[i])
        self.ep_sent[A] += sent; self.ep_lost[A] += sent - dl
        self.ep_dmax[A] = np.maximum(self.ep_dmax[A], np.where(has, pdelay, 0))
        self.ep_dsum[A] += np.where(has, pdelay, 0); self.ep_cnt[A] += has
        if t0 >= WARMUP:
            idx = np.minimum((pdelay / 0.5).astype(int), 2000)
            np.add.at(self.hist, idx, dl)
        self._util = rho; self._util[E] = 0; self._eloss = eloss
        self.util_acc += rho; self.ep_ticks += 1
        self.tick_goodput.append(dl.sum() / TICK)

    def _epoch_end(self, tnow):
        E = self.t.E; w = self.w
        A = np.array(self.active, dtype=np.int64)
        cls = w["cls"][A] if len(A) else np.array([], int)
        if len(A):
            ls = self.ep_lost[A] / np.maximum(self.ep_sent[A], 1e-12)
            sat = (ls <= LOSS_TARGET[cls]) & (self.ep_dmax[A] <= DELAY_TARGET[cls])
            S = sat.sum() / (len(A) + self.ep_rejects)
            md = self.ep_dsum[A] / np.maximum(self.ep_cnt[A], 1)
            D = np.minimum(1, md / DELAY_TARGET[cls]).mean()
            L = self.ep_lost[A].sum() / max(self.ep_sent[A].sum(), 1e-12)
            C = self.ep_disc / len(A)
        else:
            S = D = L = C = 0.0
        um = self.util_acc[:E] / max(self.ep_ticks, 1)
        U = min(1.0, float(um.max()))
        rw = self.rw
        reward = rw["S"] * S - rw["D"] * D - rw["L"] * L - rw["U"] * U - rw["C"] * C
        if self.learner is not None and self.learner.training:
            self.learner.epoch_reward(reward, self.reward_mode)
        if self.record_series:
            gp = np.mean(self.tick_goodput[-TPE:])
            self.series["t"].append(tnow); self.series["goodput"].append(gp)
            self.series["maxutil"].append(float(um.max())); self.series["churn"].append(C)
            self.series["active"].append(len(A)); self.series["loss"].append(L)
        # reset epoch accumulators
        self.ep_sent[:] = 0; self.ep_lost[:] = 0; self.ep_dmax[:] = 0; self.ep_dsum[:] = 0; self.ep_cnt[:] = 0
        self.ep_disc = 0; self.ep_rejects = 0; self.util_acc[:] = 0; self.ep_ticks = 0
        # new sending-rate draw for each active flow
        if len(A):
            self.mult[A] = self.G[A, int(round(tnow / EPOCH))]
        # discretionary rerouting (only methods that support it)
        if self.m.reroutes and len(A):
            util, _l = self.telemetry()[1], self.telemetry()[2]
            hot = util[:E] >= self.hot_thr
            if hot.any():
                hot_e = np.append(hot, False)
                onhot = [f for f in self.active if hot_e[self.path[f]].any()]
                if self.hold:
                    onhot = [f for f in onhot if self.now - self.inst_t[f] >= 2.0]
                if len(onhot) > self.max_reroute:
                    idx = self.rng.choice(len(onhot), self.max_reroute, replace=False)
                    onhot = [onhot[i] for i in sorted(idx)]
                for f in onhot:
                    self.decide(f, "reroute")

    # ------------------------------------------------------------------ results
    def finish(self):
        w = self.w; E = self.t.E
        tend = self.ep_len
        arr = w["t"]
        meas = (arr >= WARMUP) & (arr < tend - 2.0)
        idx = np.where(meas)[0]
        cls = w["cls"][idx]
        st = self.status[idx]
        served = (st == 1) | (st == 2)
        loss = 1 - self.deliv[idx] / np.maximum(self.sent[idx], 1e-12)
        p95 = np.array([np.percentile(self.dsamp[f], 95) if len(self.dsamp[f]) else 1e9 for f in idx])
        sat = served & (loss <= LOSS_TARGET[cls]) & (p95 <= DELAY_TARGET[cls])
        res = {}
        res["offered"] = int(len(idx))
        res["qos_sat"] = float(sat.mean()) if len(idx) else np.nan
        res["qos_sat_inter"] = float(sat[cls == 0].mean()) if (cls == 0).any() else np.nan
        res["qos_sat_bulk"] = float(sat[cls == 1].mean()) if (cls == 1).any() else np.nan
        res["qos_macro"] = np.nanmean([res["qos_sat_inter"], res["qos_sat_bulk"]])
        res["qos_accepted"] = float(sat[served].mean()) if served.any() else np.nan
        res["admission"] = float(((st != 3)).mean()) if len(idx) else np.nan
        res["dropped_frac"] = float((st == 4).mean()) if len(idx) else np.nan
        tot_sent = self.sent[idx][served].sum(); tot_del = self.deliv[idx][served].sum()
        res["loss"] = float(1 - tot_del / max(tot_sent, 1e-12))
        h = self.hist; cs = np.cumsum(h)
        if cs[-1] > 0:
            centers = (np.arange(len(h)) + 0.5) * 0.5
            res["delay_mean"] = float((h * centers).sum() / cs[-1])
            res["delay_p95"] = float(centers[np.searchsorted(cs, 0.95 * cs[-1])])
        else:
            res["delay_mean"] = res["delay_p95"] = np.nan
        g = np.array(self.tick_goodput); tt = np.arange(len(g)) * TICK
        res["goodput"] = float(g[tt >= WARMUP].mean())
        mu = np.array(self.series["maxutil"]); ts = np.array(self.series["t"])
        res["maxutil_mean"] = float(mu[ts > WARMUP].mean()) if len(mu) else np.nan
        ch = np.array(self.series["churn"])
        res["churn_disc"] = float(ch[ts > WARMUP].mean()) if len(ch) else 0.0
        act = np.array(self.series["active"])
        res["reroute_fail"] = self.n_reroute_fail; res["reroute_disc"] = self.n_reroute_disc
        res["reroute_fail_frac"] = self.n_reroute_fail / max(act[ts > WARMUP].mean(), 1)
        dt = np.array([d[4] for d in self.dec_times]) * 1e3 if self.dec_times else np.array([0.0])
        res["dec_ms_p50"] = float(np.percentile(dt, 50)); res["dec_ms_p95"] = float(np.percentile(dt, 95))
        res["dec_ms_p50_incl_install"] = res["dec_ms_p50"] + INSTALL * 1e3
        res["dec_cand_ms"] = float(np.mean([d[1] for d in self.dec_times]) * 1e3) if self.dec_times else 0
        res["dec_infer_ms"] = float(np.mean([d[2] for d in self.dec_times]) * 1e3) if self.dec_times else 0
        res["dec_check_ms"] = float(np.mean([d[3] for d in self.dec_times]) * 1e3) if self.dec_times else 0
        res["ctrl_cpu_s"] = self.ctrl_time
        nsw = self.t.n
        polls = int(self.ep_len / TICK)
        stats_bytes = polls * nsw * (24 + 16 + 112 * self.t.deg.mean())
        res["of_msgs"] = self.flowmods + self.barriers
        res["ctrl_bytes"] = self.flowmods * FLOWMOD_B + self.barriers * BARRIER_B + stats_bytes
        res["route_bytes"] = self.flowmods * FLOWMOD_B + self.barriers * BARRIER_B
        res["decisions"] = self.n_decisions
        res["fallback_rate"] = self.n_fallback / max(self.n_decisions, 1)
        res["masked_rate"] = self.n_masked / max(self.n_decisions, 1)
        res["rejects"] = self.n_reject; res["dropped"] = self.n_dropped
        rl = np.array(self.restore_lat, float)
        res["restore_ms_mean"] = float(np.nanmean(rl) * 1e3) if len(rl) and np.isfinite(rl).any() else np.nan
        res["restore_ms_p95"] = float(np.nanpercentile(rl, 95) * 1e3) if len(rl) and np.isfinite(rl).any() else np.nan
        res["affected"] = len(rl)
        res["fail_window_loss"] = self.fw_lost / self.fw_sent if self.fw_sent > 0 else np.nan
        self.res = res
        return res


def recovery_metrics(obs_tick_goodput, ref_tick_goodput, fail_start=60.0, fail_dur=30.0, need=4):
    """Recovery time (s) and service deficit (Mbit) against a matched warm-start reference.

    Windows are 0.25 s ticks; recovery requires `need` consecutive windows (1 s) at >=95% of the
    reference post-failure level (mean reference goodput over the failure interval)."""
    g = np.asarray(obs_tick_goodput); r = np.asarray(ref_tick_goodput)
    a = int(round(fail_start / TICK)); b = int(round((fail_start + fail_dur) / TICK))
    level = r[a:b].mean()
    ok = g[a:b] >= 0.95 * level
    rec = np.nan                                   # censored if never recovered
    for i in range(0, b - a - need + 1):
        if ok[i:i + need].all():
            rec = i * TICK                         # start of the first stable 1 s run
            break
    deficit = float(np.maximum(0, r[a:b] - g[a:b]).sum() * TICK)
    return rec, deficit, level
