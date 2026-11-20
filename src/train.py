"""Offline PPO training on the source network only; checkpoint chosen on source validation."""
import argparse, json, os, time, hashlib
import numpy as np
from topo import load, NETS
from workload import make_workload, REGIMES
from netsim import Env, PathCache, REWARD_W
from agents import Learner, LearnedMethod

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = dict(train=range(0, 600), val=range(600, 800), test=range(800, 1000))
LOADS = [0.4, 0.7, 0.95]


def split_manifest():
    m = {k: [v.start, v.stop] for k, v in SPLITS.items()}
    s = json.dumps(m, sort_keys=True)
    m["sha256"] = hashlib.sha256(s.encode()).hexdigest()
    return m


VAL_SET = [(600, 0.7, "uniform", True), (601, 0.95, "hotspot", True), (602, 0.95, "bursty", False),
           (603, 0.7, "bursty", True), (604, 0.95, "uniform", False), (605, 0.4, "hotspot", True)]


def rand_fail(topo, rng):
    return [int(rng.choice(topo.nonbridge_links))]


def validate(topo, method, K, pc):
    qs = []
    for seed, ld, reg, fail in VAL_SET:
        w = make_workload(topo, seed, ld, reg)
        fl = rand_fail(topo, np.random.default_rng(seed)) if fail else []
        r = Env(topo, w, method, K=K, fail_links=fl, pcache=pc, record_series=False).run()
        qs.append(r["qos_sat"])
    return float(np.mean(qs))


def train(src, kind="graph", seed=0, K=4, budget=100_000, fail_train=True, reward_w=None,
          reward_mode="final", hold_down=True, tag="proposed", out_dir=None, val_every=4, log=print, ln=False, lr=None):
    topo = load(src)
    L = Learner(kind, seed, K, hp=(dict(lr=lr) if lr else None), ln=ln)
    M = LearnedMethod(L).bind(topo)
    pc = PathCache(topo)
    rng = np.random.default_rng(10_000 + seed)
    out_dir = out_dir or os.path.join(ROOT, "results", "models")
    os.makedirs(out_dir, exist_ok=True)
    name = f"{tag}_{src}_s{seed}"
    best, best_ep, hist = -1, 0, []
    ep = 0; upd = 0; t0 = time.time()
    while L.n_seen + len(L.buf) < budget:
        wseed = int(rng.integers(SPLITS["train"].start, SPLITS["train"].stop))
        ld = float(rng.choice(LOADS)); reg = str(rng.choice(REGIMES))
        fl = rand_fail(topo, rng) if (fail_train and rng.random() < 0.5) else []
        w = make_workload(topo, wseed, ld, reg)
        L.training = True
        Env(topo, w, M, K=K, fail_links=fl, learner=L, reward_w=reward_w, reward_mode=reward_mode,
            hold_down=hold_down, seed=seed * 7919 + ep, pcache=pc, record_series=False).run()
        L.end_episode(); ep += 1
        if L.ready():
            st, rsum, n = L.update(M.spec); upd += 1
            if upd % val_every == 0 or L.n_seen >= budget:
                L.training = False
                v = validate(topo, M, K, pc)
                hist.append(dict(decisions=L.n_seen, val_qos=v, loss=float(st[0]), ent=float(st[3]),
                                 reward_per_dec=rsum / n, time=time.time() - t0))
                log(f"[{name}] dec={L.n_seen} val_qos={v:.4f} ent={st[3]:.3f} r/dec={rsum/n:.4f} t={time.time()-t0:.0f}s")
                if v > best:
                    best = v; best_ep = L.n_seen; L.save(os.path.join(out_dir, name + ".pkl"))
    if L.buf:
        L.buf = []
    meta = dict(name=name, src=src, kind=kind, seed=seed, K=K, budget=budget, fail_train=fail_train,
                reward_w=reward_w or REWARD_W, reward_mode=reward_mode, hold_down=hold_down,
                best_val_qos=best, best_at=best_ep, history=hist, episodes=ep,
                train_time_s=time.time() - t0)
    with open(os.path.join(out_dir, name + ".json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="Abilene"); ap.add_argument("--kind", default="graph")
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--budget", type=int, default=100000); ap.add_argument("--nofail", action="store_true")
    ap.add_argument("--nochurn", action="store_true"); ap.add_argument("--mode", default="final")
    ap.add_argument("--tag", default="proposed"); ap.add_argument("--out", default=None)
    ap.add_argument("--ln", action="store_true"); ap.add_argument("--lr", type=float, default=None)
    a = ap.parse_args()
    rw = None
    if a.nochurn:
        s = sum(v for k, v in REWARD_W.items() if k != "C")
        rw = {k: (v / s if k != "C" else 0.0) for k, v in REWARD_W.items()}
    train(a.src, a.kind, a.seed, a.K, a.budget, not a.nofail, rw, a.mode, tag=a.tag, out_dir=a.out, ln=a.ln, lr=a.lr)
