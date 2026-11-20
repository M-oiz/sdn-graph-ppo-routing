# Failure-Aware Graph PPO Routing for SDN

Code, data and results for our Computer Networks research project:

**Failure-Aware Graph Reinforcement Learning for SDN Routing: A Controlled Evaluation of Transfer to Unseen Topologies, Failure Recovery and Controller Cost**

Abdul Moiz (i24-0062) and Jawad Shahid (i24-0066), Department of Artificial Intelligence, FAST-NUCES Islamabad.

The paper is in [`paper/`](paper/).

## What is in this repository

```
src/        simulator, routing methods, graph PPO agent, training, experiments, analysis
  topo.py         load and preprocess Abilene, GEANT (2012) and NSFNET
  workload.py     max-concurrent-flow load calibration, Poisson flows, train/val/test seeds
  netsim.py       flow-level SDN simulator (reservations, queues, loss, failures, controller timing)
  methods.py      baselines B1 static shortest path, B2 ECMP, B3 load-aware heuristic
  agents.py       graph attention PPO policy (proposed) and MLP PPO (B4)
  train.py        PPO training loop (source network only)
  run_train.py    trains all 34 models
  experiments.py  main, ablation and stress experiments with frozen policies
  analysis.py     statistics, tables and figures
  test_sim.py     hand-checked simulator tests
data/       raw Topology Zoo GraphML files and the processed topology manifest (SHA-256 hashes)
results/    one JSON row per episode (main.jsonl, ablation.jsonl, stress.jsonl),
            trained checkpoints and training logs (results/models/)
report/     all figures (PDF and PNG) and LaTeX tables used in the paper
paper/      the Phase 4 research paper (PDF)
```

## How to reproduce

Python 3.13 was used. CPU only, no GPU needed.

```bash
pip install -r requirements.txt
cd src
python3 test_sim.py          # implementation checks (all should pass)
python3 run_train.py         # trains all 34 models (~60 min on 2 CPU cores)
python3 experiments.py main
python3 experiments.py ablation
python3 experiments.py stress   # full evaluation ~1.6 h on 2 CPU cores
python3 analysis.py          # writes all tables and figures to report/
```

Every random choice (workloads, traffic changes, failures, model initialisation, PPO sampling, bootstrap) uses a fixed seed. Workload seeds 0-599 are for training, 600-799 for validation and 800-999 for testing.

## Main result

| Method | QoS on training network | QoS on unseen networks |
|---|---|---|
| B1 Static shortest path | 0.891 | 0.891 |
| B2 ECMP | 0.896 | 0.896 |
| B3 Load-aware heuristic | 0.904 | 0.904 |
| B4 MLP PPO | 0.896 | 0.876 |
| Graph PPO (proposed) | 0.896 | 0.866 |

See the paper for failure recovery, controller cost, ablations and statistical tests.

## Data source

Abilene and GEANT topologies are from the Internet Topology Zoo (https://topology-zoo.org). NSFNET is the standard 14-node, 21-link backbone; its edge list is in `src/topo.py`. Traffic is synthetic.
