"""Step 4-5: graph-attention PPO routing policy (proposed) and MLP PPO (B4), in JAX."""
import os, time, pickle
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
import numpy as np
import jax, jax.numpy as jnp
from methods import k_with_prev

HID, HEADS = 64, 2
PPO = dict(lr=3e-4, gamma=0.99, lam=0.95, clip=0.2, ent=0.01, vf=0.5, rollout=2048, batch=64, epochs=10, max_grad=0.5)


# ----------------------------------------------------------------------------- observation
class GraphSpec:
    def __init__(self, topo):
        n, E = topo.n, topo.E
        self.n, self.E = n, E
        self.src = np.concatenate([topo.eu, np.arange(n)]).astype(np.int32)
        self.dst = np.concatenate([topo.ev, np.arange(n)]).astype(np.int32)
        self.degn = topo.deg / topo.deg.max()


def build_obs(env, f, cands, mask, resid, K, spec):
    t = env.t; E, n = t.E, t.n
    tel_t, util, eloss = env.tel_now
    age = min(1.0, max(0.0, env.now - tel_t)) if tel_t > -1e8 else 1.0
    cap = env.cap[:E]
    ef = np.zeros((E + n, 6), np.float32)
    ef[:E, 0] = np.clip(util[:E], 0, 1)
    ef[:E, 1] = np.clip(resid[:E] / cap, 0, 1)
    ef[:E, 2] = t.prop / 100.0
    ef[:E, 3] = np.clip(eloss[:E], 0, 1)
    ef[:E, 4] = env.avail[:E]
    ef[:E, 5] = age
    ef[E:, 4] = 1.0
    emask = np.concatenate([env.avail[:E], np.ones(n, bool)]).astype(np.float32)
    nf = np.zeros((n, 4), np.float32)
    s, d = int(env.w["src"][f]), int(env.w["dst"][f])
    nf[s, 0] = 1; nf[d, 1] = 1
    prev = env.pnodes[f] if env.status[f] == 1 else None
    if prev is not None:
        nf[list(prev), 2] = 1
    nf[:, 3] = spec.degn
    pm = np.zeros((K, n), np.float32); pf = np.zeros((K, 6), np.float32); am = np.zeros(K, np.float32)
    rate = env.w["rate"][f]
    for i, (nodes, e) in enumerate(cands[:K]):
        pm[i, list(nodes)] = 1.0 / len(nodes)
        pf[i, 0] = np.clip((resid[e] / env.cap[e]).min(), 0, 1)
        pf[i, 1] = env.prop[e].sum() / 100.0
        pf[i, 2] = 1 - np.prod(1 - np.clip(eloss[e], 0, 1))
        pf[i, 3] = len(e) / 10.0
        pf[i, 4] = 1.0 if (prev is not None and tuple(nodes) == tuple(prev)) else 0.0
        pf[i, 5] = min(1.0, rate / env.cap[e].min())
        am[i] = 1.0 if mask[i] else 0.0
    fl = np.array([float(env.w["cls"][f]), rate / 5.0], np.float32)
    g = np.array([np.clip(util[:E], 0, 1).mean(), min(1.0, util[:E].max())], np.float32)
    return dict(ef=ef, emask=emask, nf=nf, pm=pm, pf=pf, am=am, fl=fl, g=g)


# ----------------------------------------------------------------------------- models
def _dense(key, i, o, scale=None):
    scale = np.sqrt(2.0 / i) if scale is None else scale
    return dict(W=jax.random.normal(key, (i, o)) * scale, b=jnp.zeros(o))


def _lin(p, x):
    return x @ p["W"] + p["b"]


def init_graph(key, ln=False):
    ks = jax.random.split(key, 12)
    D = HID // HEADS
    def gat(k):
        k1, k2, k3, k4 = jax.random.split(k, 4)
        return dict(W=jax.random.normal(k1, (HID, HID)) * np.sqrt(1 / HID),
                    We=jax.random.normal(k2, (6, HID)) * np.sqrt(1 / 6),
                    a_s=jax.random.normal(k3, (HEADS, D)) * 0.1,
                    a_d=jax.random.normal(k4, (HEADS, D)) * 0.1)
    p = dict(inp=_dense(ks[0], 16, HID), g1=gat(ks[1]), g2=gat(ks[2]),
             a1=_dense(ks[3], HID + 8, HID), a2=_dense(ks[4], HID, 1, 0.01),
             c1=_dense(ks[5], HID + 4, HID), c2=_dense(ks[6], HID, 1, 1.0 / np.sqrt(HID)))
    if ln:
        p["ln"] = dict(g1=jnp.ones(HID), b1=jnp.zeros(HID), g2=jnp.ones(HID), b2=jnp.zeros(HID))
    return p


def _ln(x, g, b, xp=jnp):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / xp.sqrt(v + 1e-5) * g + b


def init_mlp(key):
    ks = jax.random.split(key, 6)
    return dict(a1=_dense(ks[0], 8, HID), a2=_dense(ks[1], HID, HID), a3=_dense(ks[2], HID, 1, 0.01),
                c1=_dense(ks[3], 10, HID), c2=_dense(ks[4], HID, HID), c3=_dense(ks[5], HID, 1, 1.0 / np.sqrt(HID)))


def _gat(p, h, ef, emask, src, dst, n):
    D = HID // HEADS
    z = (h @ p["W"]).reshape(n, HEADS, D)
    ze = (ef @ p["We"]).reshape(-1, HEADS, D)
    msg = z[src] + ze
    logit = jax.nn.leaky_relu((z[dst] * p["a_d"]).sum(-1) + (msg * p["a_s"]).sum(-1), 0.2)
    logit = jnp.where(emask[:, None] > 0, logit, -1e9)
    mx = jax.ops.segment_max(logit, dst, n)
    ex = jnp.exp(logit - mx[dst]) * emask[:, None]
    den = jax.ops.segment_sum(ex, dst, n)
    alpha = ex / (den[dst] + 1e-9)
    out = jax.ops.segment_sum(alpha[..., None] * msg, dst, n).reshape(n, HID)
    return jax.nn.elu(out) + h


def graph_forward(params, o, src, dst, n):
    E = src.shape[0] - n
    ef, em = o["ef"], o["emask"]
    real = em[:E][:, None]
    ins = jax.ops.segment_sum(ef[:E] * real, dst[:E], n) / jnp.maximum(jax.ops.segment_sum(real, dst[:E], n), 1)
    outs = jax.ops.segment_sum(ef[:E] * real, src[:E], n) / jnp.maximum(jax.ops.segment_sum(real, src[:E], n), 1)
    x = jnp.concatenate([o["nf"], ins, outs], -1)
    h = jnp.tanh(_lin(params["inp"], x))
    h = _gat(params["g1"], h, ef, em, src, dst, n)
    if "ln" in params:
        h = _ln(h, params["ln"]["g1"], params["ln"]["b1"])
    h = _gat(params["g2"], h, ef, em, src, dst, n)
    if "ln" in params:
        h = _ln(h, params["ln"]["g2"], params["ln"]["b2"])
    pe = o["pm"] @ h
    K = pe.shape[0]
    a_in = jnp.concatenate([pe, o["pf"], jnp.broadcast_to(o["fl"], (K, 2))], -1)
    logits = _lin(params["a2"], jnp.tanh(_lin(params["a1"], a_in)))[:, 0]
    c_in = jnp.concatenate([h.mean(0), o["fl"], o["g"]])
    v = _lin(params["c2"], jnp.tanh(_lin(params["c1"], c_in)))[0]
    return jnp.where(o["am"] > 0, logits, -1e9), v


def mlp_forward(params, o, src, dst, n):
    K = o["pf"].shape[0]
    a_in = jnp.concatenate([o["pf"], jnp.broadcast_to(o["fl"], (K, 2))], -1)
    h = jnp.tanh(_lin(params["a1"], a_in)); h = jnp.tanh(_lin(params["a2"], h))
    logits = _lin(params["a3"], h)[:, 0]
    am = o["am"]
    mpf = (o["pf"] * am[:, None]).sum(0) / jnp.maximum(am.sum(), 1)
    c = jnp.tanh(_lin(params["c1"], jnp.concatenate([mpf, o["fl"], o["g"]])))
    c = jnp.tanh(_lin(params["c2"], c))
    v = _lin(params["c3"], c)[0]
    return jnp.where(am > 0, logits, -1e9), v


# ----------------------------------------------------------------------------- learner
class Learner:
    """Holds a policy, acts inside the simulator, stores trajectories and runs PPO."""

    def __init__(self, kind, seed, K=4, hp=None, ln=False):
        self.kind = kind; self.K = K
        self.hp = dict(PPO, **(hp or {}))
        key = jax.random.PRNGKey(seed)
        self.params = init_graph(key, ln) if kind == "graph" else init_mlp(key)
        self.fwd = graph_forward if kind == "graph" else mlp_forward
        self.opt = jax.tree_util.tree_map(lambda x: (jnp.zeros_like(x), jnp.zeros_like(x)), self.params)
        self.step = 0
        self.rng = np.random.default_rng(seed + 1000)
        self.training = False
        self._jit = {}
        self.buf = []; self.ep_start = 0; self.epoch_idx = []
        self.n_seen = 0

    def spec_jit(self, spec):
        key = (spec.n, spec.E)
        if key not in self._jit:
            src, dst, n = jnp.array(spec.src), jnp.array(spec.dst), spec.n
            fwd = self.fwd
            single = jax.jit(lambda p, o: fwd(p, o, src, dst, n))
            batched = jax.vmap(lambda p, o: fwd(p, o, src, dst, n), in_axes=(None, 0))

            def loss_fn(p, ob, act, oldlp, adv, ret):
                logits, v = batched(p, ob)
                lp_all = jax.nn.log_softmax(logits)
                lp = jnp.take_along_axis(lp_all, act[:, None], 1)[:, 0]
                ratio = jnp.exp(lp - oldlp)
                c = self.hp["clip"]
                pl = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - c, 1 + c) * adv).mean()
                vl = ((v - ret) ** 2).mean()
                pr = jnp.exp(lp_all) * (ob["am"] > 0)
                ent = -(pr * jnp.where(ob["am"] > 0, lp_all, 0)).sum(-1).mean()
                return pl + self.hp["vf"] * vl - self.hp["ent"] * ent, (pl, vl, ent)

            hp = self.hp

            def update(p, opt, step, ob, act, oldlp, adv, ret):
                (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(p, ob, act, oldlp, adv, ret)
                gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
                sc = jnp.minimum(1.0, hp["max_grad"] / (gn + 1e-8))
                g = jax.tree_util.tree_map(lambda x: x * sc, g)
                b1, b2, eps = 0.9, 0.999, 1e-8
                step = step + 1
                opt = jax.tree_util.tree_map(lambda gg, mv: (b1 * mv[0] + (1 - b1) * gg, b2 * mv[1] + (1 - b2) * gg ** 2), g, opt,
                                             is_leaf=lambda x: isinstance(x, tuple))
                p = jax.tree_util.tree_map(lambda x, mv: x - hp["lr"] * (mv[0] / (1 - b1 ** step)) / (jnp.sqrt(mv[1] / (1 - b2 ** step)) + eps),
                                           p, opt, is_leaf=lambda x: isinstance(x, tuple))
                return p, opt, l, aux
            self._jit[key] = (single, jax.jit(update))
        return self._jit[key]

    # --- acting
    def act(self, spec, obs):
        single, _ = self.spec_jit(spec)
        logits, v = single(self.params, obs)
        logits = np.asarray(logits, np.float64); v = float(v)
        if self.training:
            z = logits - logits.max(); p = np.exp(z); p /= p.sum()
            a = int(self.rng.choice(len(p), p=p))
            self.buf.append(dict(obs=obs, a=a, lp=float(np.log(p[a] + 1e-12)), v=v, r=0.0, done=False))
            self.epoch_idx.append(len(self.buf) - 1)
            return a
        return int(np.argmax(logits))

    def epoch_reward(self, r, mode):
        if not self.buf:
            return
        if mode == "spread" and self.epoch_idx:
            for i in self.epoch_idx:
                self.buf[i]["r"] += r / len(self.epoch_idx)
        else:
            i = self.epoch_idx[-1] if self.epoch_idx else len(self.buf) - 1
            self.buf[i]["r"] += r
        self.epoch_idx = []

    def end_episode(self):
        if self.buf:
            self.buf[-1]["done"] = True
        self.epoch_idx = []

    def ready(self):
        return len(self.buf) >= self.hp["rollout"]

    def update(self, spec):
        _, upd = self.spec_jit(spec)
        hp = self.hp
        B = self.buf
        n = len(B)
        r = np.array([b["r"] for b in B]); v = np.array([b["v"] for b in B]); done = np.array([b["done"] for b in B])
        adv = np.zeros(n); last = 0.0
        for i in reversed(range(n)):
            nv = 0.0 if (done[i] or i == n - 1) else v[i + 1]
            nonterm = 0.0 if (done[i] or i == n - 1) else 1.0
            delta = r[i] + hp["gamma"] * nv - v[i]
            last = delta + hp["gamma"] * hp["lam"] * nonterm * last
            adv[i] = last
        ret = adv + v
        advn = (adv - adv.mean()) / (adv.std() + 1e-8)
        obs = {k: np.stack([b["obs"][k] for b in B]) for k in B[0]["obs"]}
        act = np.array([b["a"] for b in B]); lp = np.array([b["lp"] for b in B])
        stats = []
        for _ in range(hp["epochs"]):
            perm = self.rng.permutation(n)
            for s in range(0, n - hp["batch"] + 1, hp["batch"]):
                idx = perm[s:s + hp["batch"]]
                ob = {k: jnp.asarray(x[idx]) for k, x in obs.items()}
                self.params, self.opt, l, aux = upd(self.params, self.opt, self.step, ob, jnp.asarray(act[idx]),
                                                    jnp.asarray(lp[idx], jnp.float32), jnp.asarray(advn[idx], jnp.float32),
                                                    jnp.asarray(ret[idx], jnp.float32))
                self.step += 1
                stats.append([float(l)] + [float(x) for x in aux])
        self.n_seen += n
        self.buf = []
        return np.mean(stats, 0), float(r.sum()), n

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(dict(kind=self.kind, K=self.K, params=jax.tree_util.tree_map(np.asarray, self.params)), f)

    @staticmethod
    def load(path, seed=0):
        with open(path, "rb") as f:
            d = pickle.load(f)
        L = Learner(d["kind"], seed, d["K"])
        L.params = jax.tree_util.tree_map(jnp.asarray, d["params"])
        return L


class LearnedMethod:
    """Wraps a Learner as a routing method inside the simulator (same interface as baselines)."""
    reroutes = True

    def __init__(self, learner, name=None):
        self.L = learner
        self.name = name or ("Graph PPO" if learner.kind == "graph" else "B4 MLP PPO")
        self.spec = None

    def bind(self, topo):
        self.spec = GraphSpec(topo)
        return self

    def candidates(self, env, f, s, d, kind):
        return k_with_prev(env, f, s, d, env.K, kind)

    def choose(self, env, f, cands, mask, resid, kind):
        K = self.L.K
        if len(cands) > K:
            cands = cands[:K]; mask = mask[:K]
        if not mask.any():
            return None
        obs = build_obs(env, f, cands, mask, resid, K, self.spec)
        a = self.L.act(self.spec, obs)
        if not mask[a]:
            env.n_learner_invalid += 1
            return None
        return a


# ----------------------------------------------------------------------------- fast NumPy inference
# Exact NumPy replica of the JAX forward pass, used for acting (avoids per-call dispatch cost).
def _np_params(params):
    return jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), params)


class NPGraph:
    def __init__(self, spec):
        n = spec.n; Ep = len(spec.src)
        self.n = n; self.E = Ep - n
        self.src, self.dst = spec.src, spec.dst
        self.Ddst = np.zeros((n, Ep)); self.Ddst[spec.dst, np.arange(Ep)] = 1
        self.Dsrc = np.zeros((n, Ep)); self.Dsrc[spec.src, np.arange(Ep)] = 1


def _elu(x):
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def np_graph_forward(p, o, g):
    n, E = g.n, g.E
    ef, em = o["ef"].astype(np.float64), o["emask"].astype(np.float64)
    real = em[:E][:, None]
    Dd, Ds = g.Ddst[:, :E], g.Dsrc[:, :E]
    ins = (Dd @ (ef[:E] * real)) / np.maximum(Dd @ real, 1)
    outs = (Ds @ (ef[:E] * real)) / np.maximum(Ds @ real, 1)
    x = np.concatenate([o["nf"], ins, outs], -1)
    h = np.tanh(x @ p["inp"]["W"] + p["inp"]["b"])
    D = HID // HEADS
    for k in ("g1", "g2"):
        q = p[k]
        lnk = k[-1]
        z = (h @ q["W"]).reshape(n, HEADS, D)
        ze = (ef @ q["We"]).reshape(-1, HEADS, D)
        msg = z[g.src] + ze
        lg = (z[g.dst] * q["a_d"]).sum(-1) + (msg * q["a_s"]).sum(-1)
        lg = np.where(lg > 0, lg, 0.2 * lg)
        lg = np.where(em[:, None] > 0, lg, -1e9)
        mx = lg.max(0)                       # a shared constant cancels inside each node's softmax
        ex = np.exp(lg - mx) * em[:, None]
        den = g.Ddst @ ex
        al = ex / (den[g.dst] + 1e-9 * np.exp(-mx))
        out = (g.Ddst @ (al[..., None] * msg).reshape(-1, HID)).reshape(n, HID)
        h = _elu(out) + h
        if "ln" in p:
            h = _ln(h, p["ln"]["g" + lnk], p["ln"]["b" + lnk], np)
    pe = o["pm"] @ h
    K = pe.shape[0]
    a_in = np.concatenate([pe, o["pf"], np.broadcast_to(o["fl"], (K, 2))], -1)
    lo = (np.tanh(a_in @ p["a1"]["W"] + p["a1"]["b"]) @ p["a2"]["W"] + p["a2"]["b"])[:, 0]
    c_in = np.concatenate([h.mean(0), o["fl"], o["g"]])
    v = (np.tanh(c_in @ p["c1"]["W"] + p["c1"]["b"]) @ p["c2"]["W"] + p["c2"]["b"])[0]
    return np.where(o["am"] > 0, lo, -1e9), v


def np_mlp_forward(p, o, g):
    K = o["pf"].shape[0]
    a_in = np.concatenate([o["pf"], np.broadcast_to(o["fl"], (K, 2))], -1)
    h = np.tanh(a_in @ p["a1"]["W"] + p["a1"]["b"]); h = np.tanh(h @ p["a2"]["W"] + p["a2"]["b"])
    lo = (h @ p["a3"]["W"] + p["a3"]["b"])[:, 0]
    am = o["am"]
    mpf = (o["pf"] * am[:, None]).sum(0) / max(am.sum(), 1)
    c = np.tanh(np.concatenate([mpf, o["fl"], o["g"]]) @ p["c1"]["W"] + p["c1"]["b"])
    c = np.tanh(c @ p["c2"]["W"] + p["c2"]["b"])
    v = (c @ p["c3"]["W"] + p["c3"]["b"])[0]
    return np.where(am > 0, lo, -1e9), v


def _act_np(self, spec, obs):
    if getattr(self, "_np_ver", None) != self.step or getattr(self, "_np_spec", None) is not spec:
        self._np_p = _np_params(self.params); self._np_ver = self.step
        self._np_g = NPGraph(spec); self._np_spec = spec
    f = np_graph_forward if self.kind == "graph" else np_mlp_forward
    logits, v = f(self._np_p, obs, self._np_g)
    v = float(v)
    if self.training:
        z = logits - logits.max(); pr = np.exp(z); pr /= pr.sum()
        a = int(self.rng.choice(len(pr), p=pr))
        self.buf.append(dict(obs=obs, a=a, lp=float(np.log(pr[a] + 1e-12)), v=v, r=0.0, done=False))
        self.epoch_idx.append(len(self.buf) - 1)
        return a
    return int(np.argmax(logits))


Learner.act_jax = Learner.act
Learner.act = _act_np
