"""Train every learned model (main methods + ablation variants) with 2 worker processes."""
import os, sys, json, time
from multiprocessing import Pool
from train import train
from topo import NETS
from netsim import REWARD_W

BUDGET = 80_000
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "models")


def nochurn_w():
    s = sum(v for k, v in REWARD_W.items() if k != "C")
    return {k: (v / s if k != "C" else 0.0) for k, v in REWARD_W.items()}


def jobs():
    J = []
    for seed in range(3):
        for src in NETS:
            J.append(dict(src=src, kind="graph", seed=seed, tag="proposed"))
            J.append(dict(src=src, kind="mlp", seed=seed, tag="mlp"))
    for seed in range(2):
        for src in ["Abilene", "NSFNET"]:          # ablation retraining limited to two source folds (compute)
            J.append(dict(src=src, kind="graph", seed=seed, tag="nofail", fail_train=False))
            J.append(dict(src=src, kind="graph", seed=seed, tag="nochurn", reward_w=nochurn_w()))
            J.append(dict(src=src, kind="graph", seed=seed, tag="K2", K=2))
            J.append(dict(src=src, kind="graph", seed=seed, tag="K8", K=8))
    return J


def run(j):
    name = f"{j['tag']}_{j['src']}_s{j['seed']}"
    if os.path.exists(os.path.join(OUT, name + ".json")):
        return name, "skip"
    logf = open(os.path.join(OUT, name + ".log"), "w")
    def log(m):
        logf.write(m + "\n"); logf.flush()
    m = train(budget=BUDGET, out_dir=OUT, log=log, **j)
    return name, m["best_val_qos"]


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    with Pool(2) as p:
        for name, r in p.imap_unordered(run, jobs()):
            print(f"{time.time()-t0:7.0f}s done {name} best_val={r}", flush=True)
