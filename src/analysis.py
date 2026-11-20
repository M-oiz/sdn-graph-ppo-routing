"""Step 9: paired statistics, hypothesis tests and all report figures (black and grey only)."""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "report", "figs")
TAB = os.path.join(ROOT, "report", "tables")
os.makedirs(FIG, exist_ok=True); os.makedirs(TAB, exist_ok=True)

for f in glob.glob("/usr/share/fonts/truetype/liberation/LiberationSerif*.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({"font.family": "Liberation Serif", "font.size": 9, "axes.edgecolor": "black",
                     "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
                     "text.color": "black", "axes.prop_cycle": plt.cycler(color=["black"]),
                     "legend.frameon": False, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "lines.linewidth": 1.2})

NAMES = {"B1": "B1 Static SP", "B2": "B2 ECMP", "B3": "B3 Load-aware", "B4": "B4 MLP PPO", "GPPO": "Graph PPO (proposed)"}
ORDER = ["B1", "B2", "B3", "B4", "GPPO"]
STYLE = {"B1": dict(ls=":", marker="s", color="0.55"), "B2": dict(ls="-.", marker="^", color="0.55"),
         "B3": dict(ls="--", marker="o", color="0.35"), "B4": dict(ls="--", marker="D", color="black", mfc="white"),
         "GPPO": dict(ls="-", marker="o", color="black")}
HATCH = {"B1": "..", "B2": "xx", "B3": "//", "B4": "\\\\", "GPPO": ""}
FACE = {"B1": "white", "B2": "white", "B3": "0.85", "B4": "0.6", "GPPO": "0.15"}
NETS = ["Abilene", "GEANT", "NSFNET"]
RNG = np.random.default_rng(2026)
NBOOT = 10000


def read(name):
    p = os.path.join(RES, name + ".jsonl")
    rows = [json.loads(l) for l in open(p)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ data shaping
def expand_baselines(df):
    """Baselines do not depend on the fold; copy their rows into every fold where the net is evaluated."""
    b = df[df.method.isin(["B1", "B2", "B3"])]
    l = df[~df.method.isin(["B1", "B2", "B3"])]
    out = [l]
    for src in NETS:
        for net in NETS:
            x = b[b.net == net].copy()
            x["fold"] = src; x["src"] = src; x["role"] = "source" if net == src else "target"
            out.append(x)
    return pd.concat(out, ignore_index=True)


CELL = ["fold", "net", "load", "cond", "wseed"]


def cell_matrix(df, method, metric, role="target", extra=None):
    """Return dict mseed -> Series indexed by cell for one method."""
    d = df[(df.method == method) & (df.role == role)]
    if extra is not None:
        d = d[extra(d)]
    out = {}
    for ms, g in d.groupby("mseed"):
        out[ms] = g.set_index(CELL)[metric].sort_index()
    return out


def paired(df, a, b, metric, role="target", sel=None, strata=("net", "load")):
    """Hierarchical bootstrap CI and paired sign-flip permutation test for mean(a) - mean(b)."""
    A = cell_matrix(df, a, metric, role, sel); B = cell_matrix(df, b, metric, role, sel)
    idx = None
    for s in list(A.values()) + list(B.values()):
        idx = s.index if idx is None else idx.intersection(s.index)
    Am = np.stack([A[k].loc[idx].values for k in sorted(A)])
    Bm = np.stack([B[k].loc[idx].values for k in sorted(B)])
    ok = ~(np.isnan(Am).any(0) | np.isnan(Bm).any(0))
    Am, Bm = Am[:, ok], Bm[:, ok]
    idx = idx[ok]
    d = Am.mean(0) - Bm.mean(0)
    est = float(d.mean())
    # hierarchical bootstrap: resample model seeds of each method, then workload cells within strata
    cells = idx.to_frame(index=False)
    groups = cells.groupby(list(strata)).indices
    boots = np.empty(NBOOT)
    for i in range(NBOOT):
        sa = RNG.integers(0, Am.shape[0], Am.shape[0]); sb = RNG.integers(0, Bm.shape[0], Bm.shape[0])
        dd = Am[sa].mean(0) - Bm[sb].mean(0)
        pick = np.concatenate([g[RNG.integers(0, len(g), len(g))] for g in groups.values()])
        boots[i] = dd[pick].mean()
    lo, hi = np.percentile(boots, [2.5, 97.5])
    # paired permutation (sign flip) on workload-level differences
    signs = RNG.choice([-1.0, 1.0], size=(NBOOT, len(d)))
    perm = np.abs((signs * d).mean(1))
    p = float((np.sum(perm >= abs(est)) + 1) / (NBOOT + 1))
    sd = d.std(ddof=1)
    dz = est / sd if sd > 0 else np.nan
    return dict(diff=est, lo=float(lo), hi=float(hi), p=p, dz=float(dz), n=len(d),
                mean_a=float(Am.mean()), mean_b=float(Bm.mean()))


def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); m = len(ps)
    adj = np.empty(m); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def ci_mean(x, n=NBOOT):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    b = RNG.choice(x, (2000, len(x))).mean(1)
    return x.mean(), np.percentile(b, 2.5), np.percentile(b, 97.5)


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def savefig(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)


# ================================================================== tables & figures
def tex_table(name, header, rows, colspec=None):
    colspec = colspec or ("l" + "r" * (len(header) - 1))
    s = ["\\begin{tabular}{%s}" % colspec, "\\toprule", " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r == "MID":
            s.append("\\midrule"); continue
        s.append(" & ".join(str(x) for x in r) + " \\\\")
    s += ["\\bottomrule", "\\end{tabular}"]
    open(os.path.join(TAB, name + ".tex"), "w").write("\n".join(s))


def fig_training():
    fig, axs = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=False)
    base = json.load(open(os.path.join(RES, "val_baselines.json"))) if os.path.exists(os.path.join(RES, "val_baselines.json")) else {}
    for ax, src in zip(axs, NETS):
        for tag, st, lab in [("proposed", dict(color="black", ls="-"), "Graph PPO"), ("mlp", dict(color="0.45", ls="--"), "B4 MLP PPO")]:
            curves = []
            for f in sorted(glob.glob(os.path.join(RES, "models", f"{tag}_{src}_s*.json"))):
                h = json.load(open(f))["history"]
                x = [p["decisions"] / 1000 for p in h]; y = [p["val_qos"] for p in h]
                ax.plot(x, y, lw=0.6, alpha=0.6, **st)
                curves.append((x, y))
            if curves:
                xs = np.linspace(min(c[0][0] for c in curves), min(c[0][-1] for c in curves), 20)
                ys = np.mean([np.interp(xs, c[0], c[1]) for c in curves], 0)
                ax.plot(xs, ys, lw=1.8, label=lab, **st)
        if src in base:
            ax.axhline(base[src]["B3"], color="0.3", ls=":", lw=1, label="B3 Load-aware")
            ax.axhline(base[src]["B2"], color="0.6", ls="-.", lw=1, label="B2 ECMP")
        ax.set_title(f"Source: {src}"); ax.set_xlabel("Training decisions (thousands)")
    axs[0].set_ylabel("Validation QoS satisfaction")
    axs[0].legend(fontsize=7, loc="lower right")
    fig.tight_layout(); savefig(fig, "fig_training")


def method_cells(df, role="target"):
    """Mean over model seeds per cell for each method -> wide table."""
    d = df[df.role == role]
    g = d.groupby(["method"] + CELL).mean(numeric_only=True).reset_index()
    return g


def summary_table(df, role="target", sel=None, metrics=None):
    d = df[df.role == role] if role else df
    if sel is not None:
        d = d[sel(d)]
    rows = {}
    for m in ORDER:
        x = d[d.method == m]
        if len(x) == 0:
            continue
        # average model seeds per cell first
        x = x.groupby(CELL).mean(numeric_only=True)
        rows[m] = {k: x[k].mean() for k in metrics if k in x}
    return rows


def fig_architecture():
    """Implemented system: modules and data flow (boxes and arrows, black only)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.6)); ax.axis("off")
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    def box(x, y, w, h, title, lines, lw=1.0):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, lw=lw, ec="black"))
        ax.text(x + w / 2, y + h - 3.2, title, ha="center", va="top", fontsize=8.5, fontweight="bold")
        ax.text(x + w / 2, y + h - 8.4, "\n".join(lines), ha="center", va="top", fontsize=7.2, linespacing=1.25)
    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"))
    box(2, 78, 30, 20, "1  topo.py", ["Topology Zoo files + NSFNET list", "validate, map IDs, merge parallel", "bridges, link betweenness"])
    box(35, 78, 30, 20, "2  workload.py", ["max concurrent flow LP (HiGHS)", "Poisson arrivals, 3 traffic regimes", "seeded train/val/test manifests"])
    box(68, 78, 30, 20, "3  netsim.py", ["0.25 s ticks, 1 s epochs", "reservations, queues, loss", "failures, controller timing"])
    box(2, 46, 30, 24, "4  methods.py", ["B1 static shortest path", "B2 ECMP (hashed)", "B3 load-aware K-path", "shared mask + fallback"])
    box(35, 46, 30, 24, "5  agents.py", ["edge-aware graph attention", "(2 layers, 64 units, 2 heads)", "path scorer + critic, PPO", "B4: same PPO with MLP"])
    box(68, 46, 30, 24, "6  train.py", ["source network only", "80k decisions, 3 seeds", "checkpoint by source", "validation QoS"])
    box(2, 14, 46, 24, "7  experiments.py", ["frozen policies, paired workloads", "3 folds x 3 loads x 4 failure types", "ablations and stress tests", "warm-start references for recovery"])
    box(52, 14, 46, 24, "8  analysis.py", ["hierarchical bootstrap 95% CI", "paired permutation test + Holm", "tables, graphs, error groups"])
    arr(32, 88, 35, 88); arr(65, 88, 68, 88)
    arr(83, 78, 83, 70); arr(50, 78, 50, 70); arr(17, 78, 17, 70)
    arr(65, 58, 68, 58)
    arr(17, 46, 17, 38); arr(50, 46, 33, 38); arr(83, 46, 46, 38)
    arr(48, 26, 52, 26)
    ax.text(50, 4, "Every run is driven by seeds; raw per-episode rows are stored in results/*.jsonl", ha="center", fontsize=7.5, style="italic")
    savefig(fig, "fig_architecture")


if __name__ == "__main__" and False:
    pass


# ------------------------------------------------------------------ MAIN RESULTS
MAIN_METRICS = ["qos_sat", "qos_macro", "qos_sat_inter", "qos_sat_bulk", "admission", "loss", "delay_mean", "delay_p95",
                "goodput", "maxutil_mean", "churn_disc", "fallback_rate", "masked_rate", "dropped_frac",
                "dec_ms_p50", "dec_ms_p95", "dec_cand_ms", "dec_infer_ms", "dec_check_ms", "ctrl_cpu_s",
                "of_msgs", "route_bytes", "ctrl_bytes", "recovery_s", "deficit_mbit", "restore_ms_mean",
                "restore_ms_p95", "fail_window_loss", "censored", "qos_accepted", "reroute_fail_frac"]


def load_main():
    df = read("main")
    df = expand_baselines(df)
    df["censored"] = df["censored"].astype(float) if "censored" in df else np.nan
    return df


def table_main(df):
    S = summary_table(df, "target", metrics=MAIN_METRICS)
    rows = []
    for m in ORDER:
        if m not in S:
            continue
        s = S[m]
        rows.append([NAMES[m], fmt(s["qos_sat"]), fmt(s["qos_sat_inter"]), fmt(s["qos_sat_bulk"]), fmt(s["admission"]), fmt(100 * s["loss"], 2),
                     fmt(s["delay_mean"], 1), fmt(s["delay_p95"], 1), fmt(s["goodput"], 0), fmt(s["maxutil_mean"], 3),
                     fmt(100 * s["churn_disc"], 2), fmt(100 * s["fallback_rate"], 2)])
    tex_table("tab_main", ["Method", "QoS sat.", "Inter.", "Bulk", "Admit", "Loss \\%", "Delay", "Delay p95", "Goodput",
                           "Max util.", "Churn \\%", "Fallback \\%"], rows)
    return S


def table_by_net(df):
    rows = []
    for net in NETS:
        for m in ORDER:
            r = [net if m == "B1" else "", NAMES[m]]
            for ld in [0.4, 0.7, 0.95]:
                x = df[(df.role == "target") & (df.net == net) & (df.method == m) & (df.load == ld)]
                x = x.groupby(CELL).mean(numeric_only=True)
                r.append(fmt(x["qos_sat"].mean()))
            for c in ["none", "random", "targeted", "double"]:
                x = df[(df.role == "target") & (df.net == net) & (df.method == m) & (df.cond == c)]
                x = x.groupby(CELL).mean(numeric_only=True)
                r.append(fmt(x["qos_sat"].mean()))
            rows.append(r)
        rows.append("MID")
    rows = rows[:-1]
    tex_table("tab_by_net", ["Target", "Method", "40\\%", "70\\%", "95\\%", "None", "Random", "Targeted", "Double"], rows,
              "llrrrrrrr")


def table_failure(df):
    rows = []
    sel = df[(df.role == "target") & (df.cond != "none")]
    for m in ORDER:
        x = sel[sel.method == m]
        x0 = x[x.mseed == 0].groupby(CELL).mean(numeric_only=True)
        xa = x.groupby(CELL).mean(numeric_only=True)
        rec = x0["recovery_s"]; cen = x0["censored"].mean()
        rows.append([NAMES[m], fmt(xa["qos_sat"].mean()), fmt(100 * xa["fail_window_loss"].mean(), 2),
                     fmt(x0["recovery_s"].mean(), 2), fmt(100 * cen, 1), fmt(x0["deficit_mbit"].mean(), 1),
                     fmt(xa["restore_ms_mean"].mean(), 1), fmt(xa["restore_ms_p95"].mean(), 1), fmt(100 * xa["dropped_frac"].mean(), 2)])
    tex_table("tab_failure", ["Method", "QoS sat.", "Fail-window loss \\%", "Recovery (s)", "Censored \\%",
                              "Deficit (Mbit)", "Restore mean (ms)", "Restore p95 (ms)", "Dropped \\%"], rows)


def table_cost(df):
    rows = []
    sel = df[(df.role == "target")]
    for m in ORDER:
        x = sel[sel.method == m].groupby(CELL).mean(numeric_only=True)
        rows.append([NAMES[m], fmt(x["dec_ms_p50"].mean(), 3), fmt(x["dec_ms_p95"].mean(), 3), fmt(x["dec_cand_ms"].mean(), 3),
                     fmt(x["dec_infer_ms"].mean(), 3), fmt(x["dec_check_ms"].mean(), 3), fmt(x["ctrl_cpu_s"].mean(), 2),
                     fmt(x["of_msgs"].mean() / 1000, 1), fmt(x["route_bytes"].mean() / 1e6, 2), fmt(x["ctrl_bytes"].mean() / 1e6, 2)])
    tex_table("tab_cost", ["Method", "Dec. p50", "Dec. p95", "Cand.", "Infer.", "Check", "CPU (s)", "OF msgs (k)",
                           "Route MB", "Total MB"], rows)


def hyp_tests(df, abl=None):
    out = []
    # H1: graph vs MLP on held-out target QoS
    out.append(("H1", "Graph PPO $-$ B4 MLP PPO, target QoS (pp)", paired(df, "GPPO", "B4", "qos_sat"), 100))
    # transfer degradation per fold/target/load/cond
    out.append(("--", "Graph PPO $-$ B3, target QoS (pp)", paired(df, "GPPO", "B3", "qos_sat"), 100))
    out.append(("--", "Graph PPO $-$ B1, target QoS (pp)", paired(df, "GPPO", "B1", "qos_sat"), 100))
    out.append(("--", "Graph PPO $-$ B2, target QoS (pp)", paired(df, "GPPO", "B2", "qos_sat"), 100))
    return out


def degradation(df, method, metric="qos_sat"):
    d = df[df.method == method]
    g = d.groupby(["fold", "role", "net", "load", "cond", "mseed"])[metric].mean().reset_index()
    src = g[g.role == "source"].set_index(["fold", "load", "cond", "mseed"])[metric]
    tg = g[g.role == "target"].copy()
    tg["src_val"] = [src.loc[(r.fold, r.load, r.cond, r.mseed)] for r in tg.itertuples()]
    tg["deg"] = tg["src_val"] - tg[metric]
    return tg


def deg_test(df, a="GPPO", b="B4"):
    A = degradation(df, a).groupby(["fold", "net", "load", "cond"])["deg"].mean()
    B = degradation(df, b).groupby(["fold", "net", "load", "cond"])["deg"].mean()
    d = (A - B.loc[A.index]).values
    boots = np.array([RNG.choice(d, len(d)).mean() for _ in range(NBOOT)])
    signs = RNG.choice([-1.0, 1.0], size=(NBOOT, len(d)))
    p = float((np.sum(np.abs((signs * d).mean(1)) >= abs(d.mean())) + 1) / (NBOOT + 1))
    return dict(diff=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), p=p,
                n=len(d), mean_a=float(A.mean()), mean_b=float(B.mean()), dz=float(d.mean() / d.std(ddof=1)))


def fig_qos_load(df):
    fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.3), sharey=True)
    for ax, c, title in zip(axs, ["none", "random", "targeted", "double"], ["No failure", "Random single", "Targeted single", "Double"]):
        for m in ORDER:
            ys, lo, hi = [], [], []
            for ld in [0.4, 0.7, 0.95]:
                x = df[(df.role == "target") & (df.method == m) & (df.cond == c) & (df.load == ld)]
                x = x.groupby(CELL)["qos_sat"].mean()
                mu, l, h = ci_mean(x.values)
                ys.append(mu); lo.append(mu - l); hi.append(h - mu)
            st = STYLE[m]
            ax.errorbar([40, 70, 95], ys, yerr=[lo, hi], label=NAMES[m], capsize=2, ms=3.5, lw=1.0, elinewidth=0.6, **st)
        ax.set_title(title); ax.set_xticks([40, 70, 95]); ax.set_xlabel("Offered load (%)")
    axs[0].set_ylabel("QoS satisfaction (target nets)")
    axs[0].legend(fontsize=6.5, loc="lower left")
    fig.tight_layout(); savefig(fig, "fig_qos_load")


def fig_transfer(df):
    fig, ax = plt.subplots(figsize=(6.0, 2.4))
    xs = np.arange(3); w = 0.19
    items = [("GPPO", "source", "Graph PPO, source test", "0.15", ""), ("GPPO", "target", "Graph PPO, target test", "white", ""),
             ("B4", "source", "B4 MLP PPO, source test", "0.6", ""), ("B4", "target", "B4 MLP PPO, target test", "white", "////")]
    for i, (m, role, lab, fc, hatch) in enumerate(items):
        vals, errs = [], []
        for src in NETS:
            x = df[(df.method == m) & (df.role == role) & (df.fold == src)].groupby(CELL)["qos_sat"].mean()
            mu, l, h = ci_mean(x.values); vals.append(mu); errs.append([mu - l, h - mu])
        ax.bar(xs + (i - 1.5) * w, vals, w, yerr=np.array(errs).T, label=lab, color=fc, edgecolor="black", hatch=hatch,
               lw=0.7, capsize=2, error_kw=dict(lw=0.6))
    ax.set_xticks(xs); ax.set_xticklabels([f"Trained on {s}" for s in NETS])
    ax.set_ylabel("QoS satisfaction"); ax.set_ylim(0.7, 1.0)
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    fig.tight_layout(); savefig(fig, "fig_transfer")


def fig_recovery(df):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
    for ax, ld in zip(axs, [0.7, 0.95]):
        for m in ORDER:
            x = df[(df.method == m) & (df.role == "target") & (df.cond == "targeted") & (df.load == ld) & (df.mseed == 0)]
            x = x[x["gp_series"].notna()] if "gp_series" in x else x.iloc[0:0]
            if len(x) == 0:
                continue
            S = np.array([np.array(s) / lv for s, lv in zip(x["gp_series"], x["ref_level"])])
            t = np.arange(S.shape[1]) * 0.25
            sel = (t >= 56) & (t <= 66)
            ax.plot(t[sel], S.mean(0)[sel], label=NAMES[m], **{k: v for k, v in STYLE[m].items() if k != "marker"})
        ax.axvline(60, color="0.5", lw=0.6, ls=":")
        ax.axhline(0.95, color="0.5", lw=0.6, ls=":")
        ax.set_title(f"Targeted failure at 60 s, load {int(ld*100)}%")
        ax.set_xlabel("Time (s)")
    axs[0].set_ylabel("Goodput / attainable level")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout(); savefig(fig, "fig_recovery")


def fig_pareto(df):
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for m in ORDER:
        x = df[(df.role == "target") & (df.method == m)].groupby(CELL).mean(numeric_only=True)
        q = x["qos_sat"].mean()
        st = STYLE[m]
        for ax, col, scale in [(axs[0], "dec_ms_p50", 1), (axs[1], "route_bytes", 1e-6)]:
            ax.plot(x[col].mean() * scale, q, marker=st["marker"], color=st["color"], mfc=st.get("mfc", st["color"]),
                    ms=6, ls="none", label=NAMES[m])
            ax.annotate(m if m != "GPPO" else "Graph PPO", (x[col].mean() * scale, q), textcoords="offset points",
                        xytext=(5, 3), fontsize=7)
    axs[0].set_xscale("log"); axs[0].set_xlabel("Median decision time per flow (ms, log scale)")
    axs[1].set_xlabel("Route-installation control traffic per episode (MB)")
    axs[0].set_ylabel("QoS satisfaction (target nets)")
    fig.tight_layout(); savefig(fig, "fig_pareto")


def fig_outcomes(df, ld=0.95):
    fig, ax = plt.subplots(figsize=(6.0, 2.3))
    cats = [("satisfied", "white", ""), ("served, QoS missed", "0.75", ""), ("rejected at admission", "0.35", ""),
            ("dropped after failure", "black", "")]
    left = np.zeros(len(ORDER))
    vals = []
    for m in ORDER:
        x = df[(df.role == "target") & (df.method == m) & (df.load == ld)].groupby(CELL).mean(numeric_only=True)
        sat = x["qos_sat"].mean(); rej = 1 - x["admission"].mean(); drop = x["dropped_frac"].mean()
        vals.append([sat, 1 - sat - rej - drop, rej, drop])
    vals = np.array(vals)
    y = np.arange(len(ORDER))
    for j, (lab, fc, h) in enumerate(cats):
        ax.barh(y, vals[:, j], left=left, color=fc, edgecolor="black", lw=0.6, label=lab, height=0.6)
        left += vals[:, j]
    ax.set_yticks(y); ax.set_yticklabels([NAMES[m] for m in ORDER]); ax.invert_yaxis()
    ax.set_xlim(0.6, 1.0); ax.set_xlabel(f"Share of offered flows (target nets, {int(ld*100)}% load)")
    ax.legend(fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.45, 1.22))
    fig.tight_layout(); savefig(fig, "fig_outcomes")
    return vals


# ------------------------------------------------------------------ ABLATIONS
ABL_ORDER = ["no graph relations (B4 MLP)", "no failure training", "no churn penalty", "no hold-down",
             "no fallback", "K = 2", "K = 8"]
VARMAP = {"proposed-no-failure-training": "no failure training", "no-churn-penalty": "no churn penalty",
          "K=2": "K = 2", "K=8": "K = 8", "no-hold-down": "no hold-down", "no-fallback": "no fallback"}


def load_ablation(main):
    ab = read("ablation")
    ab["method"] = ab["variant"].map(VARMAP)
    cells = ab[CELL].drop_duplicates()
    key = cells.set_index(CELL).index
    base = main[(main.role == "target") & (main.mseed < 2) & main.method.isin(["GPPO", "B4"])].copy()
    base = base[base.set_index(CELL).index.isin(key)]
    base["method"] = base["method"].map({"GPPO": "full", "B4": "no graph relations (B4 MLP)"})
    b3 = main[(main.role == "target") & (main.method == "B3")].copy()
    b3 = b3[b3.set_index(CELL).index.isin(key)]
    b3["method"] = "B3"
    out = pd.concat([ab, base, b3], ignore_index=True)
    out["role"] = "target"
    return out


def table_ablation(ab):
    rows, res = [], {}
    full = ab[ab.method == "full"].groupby(CELL).mean(numeric_only=True)
    rows.append(["Full graph PPO (reference)", fmt(full["qos_sat"].mean()), "--", fmt(100 * full["churn_disc"].mean(), 2),
                 fmt(100 * full["fail_window_loss"].mean(), 2), fmt(100 * full["fallback_rate"].mean(), 2),
                 fmt(full["dec_ms_p50"].mean(), 3)])
    for v in ABL_ORDER:
        if v not in set(ab.method):
            continue
        r = paired(ab, v, "full", "qos_sat")
        c = paired(ab, v, "full", "churn_disc")
        fw = paired(ab, v, "full", "fail_window_loss", sel=lambda d: d.cond != "none")
        x = ab[ab.method == v].groupby(CELL).mean(numeric_only=True)
        res[v] = dict(q=r, churn=c, fw=fw)
        rows.append([v, fmt(r["mean_a"]), f"{100*r['diff']:+.2f} [{100*r['lo']:+.2f}, {100*r['hi']:+.2f}]",
                     fmt(100 * x["churn_disc"].mean(), 2), fmt(100 * x["fail_window_loss"].mean(), 2),
                     fmt(100 * x["fallback_rate"].mean(), 2), fmt(x["dec_ms_p50"].mean(), 3)])
    tex_table("tab_ablation", ["Variant", "QoS sat.", "$\\Delta$ QoS vs full (pp) [95\\% CI]", "Churn \\%",
                               "Fail-window loss \\%", "Fallback \\%", "Dec. p50 (ms)"], rows, "lrrrrrr")
    return res


def fig_ablation(res):
    fig, ax = plt.subplots(figsize=(5.8, 2.4))
    names = [v for v in ABL_ORDER if v in res]
    y = np.arange(len(names))
    d = np.array([100 * res[v]["q"]["diff"] for v in names])
    lo = np.array([100 * res[v]["q"]["lo"] for v in names]); hi = np.array([100 * res[v]["q"]["hi"] for v in names])
    ax.barh(y, d, xerr=[d - lo, hi - d], color="0.6", edgecolor="black", lw=0.6, height=0.55, capsize=2, error_kw=dict(lw=0.7))
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlabel("Change in target QoS satisfaction vs full graph PPO (percentage points)")
    fig.tight_layout(); savefig(fig, "fig_ablation")


def fig_k(ab):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    ks = [("K = 2", 2), ("full", 4), ("K = 8", 8)]
    q, t = [], []
    for v, k in ks:
        x = ab[ab.method == v].groupby(CELL).mean(numeric_only=True)
        q.append(x["qos_sat"].mean()); t.append(x["dec_ms_p50"].mean())
    ax.plot([2, 4, 8], q, "o-", color="black", label="QoS satisfaction")
    ax.set_xlabel("Number of candidate paths K"); ax.set_ylabel("QoS satisfaction"); ax.set_xticks([2, 4, 8])
    ax2 = ax.twinx(); ax2.plot([2, 4, 8], t, "s--", color="0.5", mfc="white", label="decision time")
    ax2.set_ylabel("Median decision time (ms)"); ax2.spines["right"].set_visible(True)
    l1, b1 = ax.get_legend_handles_labels(); l2, b2 = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, b1 + b2, fontsize=6.5, loc="lower right")
    fig.tight_layout(); savefig(fig, "fig_k")
    return q, t


# ------------------------------------------------------------------ STRESS
STRESS_ORDER = ["telemetry 0 ms", "telemetry +250 ms", "telemetry +1000 ms", "5% missing polls",
                "heterogeneous 50-200 Mbit/s", "burst + single failure"]


def load_stress():
    st = read("stress")
    b = st[st.method.isin(["B1", "B2", "B3"])]
    l = st[~st.method.isin(["B1", "B2", "B3"])]
    out = [l]
    for src in NETS:
        for net in NETS:
            if net == src:
                continue
            x = b[b.net == net].copy(); x["fold"] = src; x["src"] = src; x["role"] = "target"; out.append(x)
    return pd.concat(out, ignore_index=True)


def table_stress(st):
    rows = []
    vals = {}
    for m in ORDER:
        r = [NAMES[m]]
        for s in STRESS_ORDER:
            x = st[(st.method == m) & (st.stress == s)].groupby(CELL + ["stress"]).mean(numeric_only=True)
            v = x["qos_sat"].mean(); vals[(m, s)] = v
            r.append(fmt(v))
        rows.append(r)
    tex_table("tab_stress", ["Method", "Telemetry 0 ms", "+250 ms", "+1000 ms", "5\\% missing", "Hetero. cap.", "Burst + fail"],
              rows)
    rows2 = []
    for m in ORDER:
        r = [NAMES[m]]
        for s in STRESS_ORDER:
            x = st[(st.method == m) & (st.stress == s)].groupby(CELL + ["stress"]).mean(numeric_only=True)
            r.append(fmt(x["recovery_s"].mean(), 2))
        rows2.append(r)
    tex_table("tab_stress_rec", ["Method", "Telemetry 0 ms", "+250 ms", "+1000 ms", "5\\% missing", "Hetero. cap.", "Burst + fail"],
              rows2)
    return vals


def fig_stress(vals):
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    xs = np.arange(len(STRESS_ORDER)); w = 0.16
    for i, m in enumerate(ORDER):
        v = [vals[(m, s)] for s in STRESS_ORDER]
        ax.bar(xs + (i - 2) * w, v, w, color=FACE[m], edgecolor="black", hatch=HATCH[m], lw=0.6, label=NAMES[m])
    ax.set_xticks(xs); ax.set_xticklabels(["Telemetry\n0 ms", "Telemetry\n+250 ms", "Telemetry\n+1000 ms", "5% missing\npolls",
                                           "Heterogeneous\ncapacity", "Burst +\nfailure"], fontsize=7.5)
    lo = min(vals.values()); ax.set_ylim(max(0, lo - 0.05), min(1, max(vals.values()) + 0.03))
    ax.set_ylabel("QoS satisfaction (target nets)")
    ax.legend(fontsize=6.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout(); savefig(fig, "fig_stress")


def rel_vs_b1(df, m, net, metric, sel=None):
    d = df[(df.role == "target") & (df.net == net)]
    if sel is not None:
        d = d[sel(d)]
    a = d[d.method == m].groupby(CELL)[metric].mean()
    b = d[d.method == "B1"].groupby(CELL)[metric].mean().loc[a.index]
    return 100 * (a.mean() - b.mean()) / b.mean()


def table_compare(df):
    """Our relative change against the shortest-path baseline, for the metrics the papers report."""
    out = {}
    for net in NETS:
        out[net] = {m: dict(delay=rel_vs_b1(df, m, net, "delay_mean"), loss=rel_vs_b1(df, m, net, "loss"),
                            mlu=rel_vs_b1(df, m, net, "maxutil_mean"), qos=rel_vs_b1(df, m, net, "qos_sat"),
                            fwl=rel_vs_b1(df, m, net, "fail_window_loss", lambda d: d.cond != "none"))
                    for m in ["B3", "B4", "GPPO"]}
    return out


def table_fold(df):
    rows = []
    for src in NETS:
        r = [src]
        for m in ORDER:
            x = df[(df.fold == src) & (df.role == "target") & (df.method == m)].groupby(CELL)["qos_sat"].mean()
            r.append(fmt(x.mean()))
        for m in ["B4", "GPPO"]:
            for ms in range(3):
                x = df[(df.fold == src) & (df.role == "target") & (df.method == m) & (df.mseed == ms)]["qos_sat"]
                r.append(fmt(x.mean()))
        rows.append(r)
    tex_table("tab_fold", ["Source (fold)", "B1", "B2", "B3", "B4", "GPPO", "B4 s0", "B4 s1", "B4 s2", "GPPO s0", "GPPO s1", "GPPO s2"], rows)
