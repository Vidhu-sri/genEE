"""
plot_results.py — Generate all paper figures from experiment logs.

Reads the structured output from runner.py:
  results/<run_id>/results.csv       — flat CSV for quick aggregation
  results/<run_id>/logs/*.json       — rich per-iteration JSON
  results/<run_id>/run_config.json   — run metadata

Usage:
  python plot_results.py                          # all plots, all data
  python plot_results.py --plot ctr_curves        # just CTR curves
  python plot_results.py --domain ecommerce       # filter by domain
  python plot_results.py --results-dir ./results --output-dir ./figures
"""

import json, argparse, warnings
from pathlib import Path
from collections import defaultdict
from typing import List

import numpy as np
import pandas as pd

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("pip install matplotlib")

try:
    import seaborn as sns; HAS_SNS = True
except ImportError:
    HAS_SNS = False

warnings.filterwarnings("ignore")

# ─── Style ───
COLORS = {
    "no_drop": "#636363", "random_ctr": "#3182bd",
    "partial_ctr": "#e6550d", "full_ctr": "#31a354",
    "explore_exploit": "#756bb1",
    "bandit_ucb": "#e7298a", "bandit_thompson": "#d95f02", "bandit_epsilon": "#7570b3",
}
LABELS = {
    "no_drop": "NO-DROP", "random_ctr": "RANDOM-CTR",
    "partial_ctr": "PARTIAL-CTR", "full_ctr": "FULL-CTR",
    "explore_exploit": "EXPLORE-EXPLOIT",
    "bandit_ucb": "BANDIT-UCB", "bandit_thompson": "BANDIT-TS",
    "bandit_epsilon": "BANDIT-ε",
}

def setup():
    if not HAS_MPL: return
    plt.rcParams.update({
        "figure.figsize": (10,6), "font.size": 12, "figure.dpi": 150,
        "savefig.dpi": 300, "savefig.bbox": "tight",
    })
    if HAS_SNS: sns.set_style("whitegrid")

# ─── Data loading ───

def discover_runs(base: Path) -> List[dict]:
    runs = []
    for d in sorted(base.iterdir()):
        cfg_file = d / "run_config.json"
        if d.is_dir() and cfg_file.exists():
            c = json.loads(cfg_file.read_text())
            c["_dir"] = str(d)
            runs.append(c)
    return runs

def load_all_csv(base: Path) -> pd.DataFrame:
    dfs = []
    for r in discover_runs(base):
        csv = Path(r["_dir"]) / "results.csv"
        if csv.exists():
            try:
                df = pd.read_csv(csv)
                dfs.append(df)
            except Exception as e:
                print(f"  warn: {csv}: {e}")
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def load_logs(run_dir: Path, topic: str) -> List[dict]:
    ld = Path(run_dir) / "logs"
    if not ld.exists(): return []
    logs = []
    for f in sorted(ld.glob(f"{topic}_iter_*.json")):
        logs.append(json.loads(f.read_text()))
    return sorted(logs, key=lambda x: x["iteration"])


# ═══════════════════════════════════════════════
# PLOT 1: CTR over iterations — Figure 4 replica
# ═══════════════════════════════════════════════

def plot_ctr_curves(df, domain, out):
    """One line per method, averaged across topics. ±1 std band."""
    if not HAS_MPL: return
    sub = df[df["domain"] == domain]
    if sub.empty: print(f"  no data for {domain}"); return

    fig, ax = plt.subplots()
    for method in sorted(sub["method"].unique()):
        ms = sub[sub["method"] == method]
        g = ms.groupby("iteration")["avg_ctr"].agg(["mean","std"]).reset_index()
        c = COLORS.get(method, "#333")
        ax.plot(g["iteration"], g["mean"]*100, color=c, label=LABELS.get(method,method),
                linewidth=2, marker="o", markersize=4)
        ax.fill_between(g["iteration"], (g["mean"]-g["std"])*100,
                        (g["mean"]+g["std"])*100, alpha=0.12, color=c)

    ax.set_xlabel("Iteration"); ax.set_ylabel("CTR (%)")
    ax.set_title(f"CTR over Iterations — {domain.title()}")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    p = out / f"fig4_ctr_curves_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 2: QS over iterations (average persona scores)
# ═══════════════════════════════════════════════

def plot_qs_curves(df, base, domain, out):
    """Average question relevance score per iteration per method."""
    if not HAS_MPL: return
    runs = [r for r in discover_runs(base) if r.get("domain") == domain]
    if not runs: return

    # Collect: method -> iteration -> [avg_score_across_personas]
    data = defaultdict(lambda: defaultdict(list))
    for r in runs:
        method = r["method"]
        for topic in r.get("topics", []):
            for log in load_logs(r["_dir"], topic):
                it = log["iteration"]
                pavgs = log.get("persona_avg_scores", {})
                if pavgs:
                    data[method][it].append(np.mean(list(pavgs.values())))

    fig, ax = plt.subplots()
    for method in sorted(data.keys()):
        iters = sorted(data[method].keys())
        means = [np.mean(data[method][i]) for i in iters]
        stds = [np.std(data[method][i]) for i in iters]
        c = COLORS.get(method, "#333")
        ax.plot(iters, means, color=c, label=LABELS.get(method,method), linewidth=2, marker="o", markersize=4)
        ax.fill_between(iters, np.array(means)-np.array(stds),
                        np.array(means)+np.array(stds), alpha=0.12, color=c)

    ax.set_xlabel("Iteration"); ax.set_ylabel("Avg Question Score (1-10)")
    ax.set_title(f"Question Scores over Iterations — {domain.title()}")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    p = out / f"fig4_qs_curves_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 3: Per-persona QS + CTR (Figure 12 replica)
# ═══════════════════════════════════════════════

def plot_per_persona(df, base, domain, out):
    if not HAS_MPL: return
    runs = [r for r in discover_runs(base) if r.get("domain") == domain]
    if not runs: return

    # {method -> persona -> iteration -> [score]}
    pdata = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in runs:
        m = r["method"]
        for topic in r.get("topics", []):
            for log in load_logs(r["_dir"], topic):
                it = log["iteration"]
                for persona, avg in log.get("persona_avg_scores", {}).items():
                    pdata[m][persona][it].append(avg)

    all_personas = sorted({p for m in pdata for p in pdata[m]})
    if not all_personas: return

    n = len(all_personas)
    fig, axes = plt.subplots(n, 2, figsize=(14, 3.5*n), squeeze=False)

    for pi, persona in enumerate(all_personas):
        ax_qs, ax_ctr = axes[pi, 0], axes[pi, 1]
        for method in sorted(pdata.keys()):
            pd_ = pdata[method].get(persona, {})
            if not pd_: continue
            iters = sorted(pd_.keys())
            means = [np.mean(pd_[i]) for i in iters]
            c = COLORS.get(method, "#333")
            ax_qs.plot(iters, means, color=c, label=LABELS.get(method,method), linewidth=1.5, marker="o", markersize=3)
        ax_qs.set_ylabel("QS"); ax_qs.set_title(persona); ax_qs.grid(True, alpha=0.3)
        if pi == 0: ax_qs.legend(fontsize=7)

        # CTR subplot from csv
        sub = df[(df["domain"]==domain)]
        for method in sorted(sub["method"].unique()):
            ms = sub[sub["method"]==method]
            g = ms.groupby("iteration")["avg_ctr"].mean().reset_index()
            c = COLORS.get(method, "#333")
            ax_ctr.plot(g["iteration"], g["avg_ctr"]*100, color=c, linewidth=1.5, marker="o", markersize=3)
        ax_ctr.set_ylabel("CTR (%)"); ax_ctr.set_title(persona); ax_ctr.grid(True, alpha=0.3)

    axes[-1,0].set_xlabel("Iteration"); axes[-1,1].set_xlabel("Iteration")
    fig.suptitle(f"Per-Persona Results — {domain.title()}", fontsize=14, y=1.01)
    fig.tight_layout()
    p = out / f"fig12_per_persona_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 4: Generator model comparison
# ═══════════════════════════════════════════════

def plot_model_cmp(df, domain, out):
    if not HAS_MPL: return
    sub = df[(df["domain"]==domain) & (df["method"]=="explore_exploit")]
    if sub.empty or "generator_model" not in sub.columns: return
    models = sub["generator_model"].unique()
    if len(models) <= 1: print("  only 1 generator, skip model cmp"); return

    fig, ax = plt.subplots()
    colors = plt.cm.Set1(np.linspace(0,1,len(models)))
    for mi, model in enumerate(sorted(models)):
        ms = sub[sub["generator_model"]==model]
        g = ms.groupby("iteration")["avg_ctr"].agg(["mean","std"]).reset_index()
        ax.plot(g["iteration"], g["mean"]*100, color=colors[mi], label=model, linewidth=2, marker="o", markersize=5)
        ax.fill_between(g["iteration"], (g["mean"]-g["std"])*100, (g["mean"]+g["std"])*100, alpha=0.12, color=colors[mi])

    ax.set_xlabel("Iteration"); ax.set_ylabel("CTR (%)")
    ax.set_title(f"Generator Model Comparison — {domain.title()}")
    ax.legend(); ax.grid(True, alpha=0.3)
    p = out / f"model_cmp_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 5: Evaluator comparison (GPT-4 vs FiLM)
# ═══════════════════════════════════════════════

def plot_eval_cmp(df, domain, out):
    if not HAS_MPL: return
    sub = df[(df["domain"]==domain) & (df["method"]=="explore_exploit")]
    if sub.empty or "evaluator" not in sub.columns: return
    evals = sub["evaluator"].unique()
    if len(evals) <= 1: print("  only 1 evaluator, skip eval cmp"); return

    fig, ax = plt.subplots()
    for ei, ev in enumerate(sorted(evals)):
        es = sub[sub["evaluator"]==ev]
        g = es.groupby("iteration")["avg_ctr"].agg(["mean","std"]).reset_index()
        ax.plot(g["iteration"], g["mean"]*100, label=f"Eval: {ev}", linewidth=2, marker="o", markersize=4)

    ax.set_xlabel("Iteration"); ax.set_ylabel("CTR (%)")
    ax.set_title(f"Evaluator Comparison — {domain.title()}")
    ax.legend(); ax.grid(True, alpha=0.3)
    p = out / f"eval_cmp_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 6: Cumulative regret
# ═══════════════════════════════════════════════

def plot_regret(df, domain, out):
    if not HAS_MPL: return
    sub = df[df["domain"]==domain]
    if sub.empty: return

    oracle = sub.groupby(["topic","iteration"])["avg_ctr"].max().reset_index().rename(columns={"avg_ctr":"oracle"})
    sub = sub.merge(oracle, on=["topic","iteration"])
    sub["regret"] = sub["oracle"] - sub["avg_ctr"]

    fig, ax = plt.subplots()
    for method in sorted(sub["method"].unique()):
        ms = sub[sub["method"]==method]
        g = ms.groupby("iteration")["regret"].mean().reset_index()
        g["cum_regret"] = g["regret"].cumsum()
        c = COLORS.get(method, "#333")
        ax.plot(g["iteration"], g["cum_regret"]*100, color=c, label=LABELS.get(method,method), linewidth=2)

    ax.set_xlabel("Iteration"); ax.set_ylabel("Cumulative Regret (CTR %pts)")
    ax.set_title(f"Cumulative Regret — {domain.title()}")
    ax.legend(); ax.grid(True, alpha=0.3)
    p = out / f"regret_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 7: Time breakdown per method
# ═══════════════════════════════════════════════

def plot_cost(df, out):
    if not HAS_MPL: return
    time_cols = [c for c in ["eval_time_sec","gen_time_sec","sim_time_sec"] if c in df.columns]
    if not time_cols: return

    g = df.groupby("method")[time_cols].sum().reset_index()

    fig, ax = plt.subplots()
    x = np.arange(len(g)); w = 0.25
    for ci, col in enumerate(time_cols):
        ax.bar(x + ci*w, g[col], w, label=col.replace("_sec","").replace("_"," ").title())
    ax.set_xticks(x + w); ax.set_xticklabels([LABELS.get(m,m) for m in g["method"]], rotation=25, ha="right")
    ax.set_ylabel("Total Time (s)"); ax.set_title("Computation Time Breakdown")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    p = out / "cost_breakdown.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 8: Score distributions & bias
# ═══════════════════════════════════════════════

def plot_scores(base, domain, out):
    if not HAS_MPL: return
    runs = [r for r in discover_runs(base) if r.get("domain")==domain]
    rows = []
    for r in runs:
        for topic in r.get("topics", []):
            for log in load_logs(r["_dir"], topic):
                for q in log.get("questions", []):
                    for p, s in q.get("persona_scores", {}).items():
                        rows.append({"persona": p, "score": s, "ctr": q["ctr"],
                                     "iteration": log["iteration"], "method": log["method"]})
    if not rows: print(f"  no score data for {domain}"); return
    sdf = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Distribution per persona
    for persona in sorted(sdf["persona"].unique()):
        axes[0].hist(sdf[sdf["persona"]==persona]["score"], bins=20, alpha=0.4, label=persona, density=True)
    axes[0].set_xlabel("Score"); axes[0].set_ylabel("Density"); axes[0].set_title("Score Dist by Persona")
    axes[0].legend(fontsize=7)

    # Score vs CTR
    qa = sdf.groupby("ctr").agg({"score":"mean"}).reset_index()
    axes[1].scatter(qa["score"], qa["ctr"]*100, alpha=0.3, s=10)
    if len(qa)>2:
        r = qa["score"].corr(qa["ctr"])
        axes[1].text(0.05, 0.95, f"r={r:.3f}", transform=axes[1].transAxes, fontsize=12, va="top")
    axes[1].set_xlabel("Avg Score"); axes[1].set_ylabel("CTR (%)"); axes[1].set_title("Score vs CTR")

    # Score variance over iterations
    iv = sdf.groupby(["iteration","method"])["score"].std().reset_index()
    for method in iv["method"].unique():
        ms = iv[iv["method"]==method]
        axes[2].plot(ms["iteration"], ms["score"], color=COLORS.get(method,"#333"), label=LABELS.get(method,method))
    axes[2].set_xlabel("Iteration"); axes[2].set_ylabel("Score Std"); axes[2].set_title("Score Variance")
    axes[2].legend(fontsize=7)

    fig.suptitle(f"Evaluator Analysis — {domain.title()}", fontsize=14); fig.tight_layout()
    p = out / f"scores_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 9: Topic × Method heatmap
# ═══════════════════════════════════════════════

def plot_heatmap(df, domain, out):
    if not (HAS_MPL and HAS_SNS): return
    sub = df[df["domain"]==domain]
    if sub.empty: return
    mx = sub.groupby(["topic","method"])["iteration"].max().reset_index()
    final = sub.merge(mx, on=["topic","method","iteration"])
    piv = final.pivot_table(values="avg_ctr", index="topic", columns="method", aggfunc="mean") * 100
    piv.columns = [LABELS.get(c,c) for c in piv.columns]
    if piv.empty: return

    fig, ax = plt.subplots(figsize=(12, max(8, len(piv)*0.4)))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title(f"Final CTR (%) — {domain.title()}"); ax.set_ylabel("Topic")
    p = out / f"heatmap_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# PLOT 10: Convergence speed
# ═══════════════════════════════════════════════

def plot_convergence(df, domain, out):
    if not HAS_MPL: return
    sub = df[df["domain"]==domain]
    if sub.empty: return

    thresholds = [0.8, 0.9, 0.95]
    conv = defaultdict(lambda: defaultdict(list))

    for method in sub["method"].unique():
        ms = sub[sub["method"]==method]
        for topic in ms["topic"].unique():
            ts = ms[ms["topic"]==topic].sort_values("iteration")
            if ts.empty: continue
            final = ts["avg_ctr"].iloc[-1]
            if final <= 0: continue
            for th in thresholds:
                reached = ts[ts["avg_ctr"] >= final*th]
                conv[method][th].append(reached["iteration"].iloc[0] if len(reached) else ts["iteration"].max())

    fig, ax = plt.subplots()
    methods = sorted(conv.keys()); x = np.arange(len(methods)); w = 0.25
    for ti, th in enumerate(thresholds):
        means = [np.mean(conv[m][th]) if conv[m][th] else 0 for m in methods]
        ax.bar(x + ti*w, means, w, label=f"{int(th*100)}% of final")
    ax.set_xticks(x+w); ax.set_xticklabels([LABELS.get(m,m) for m in methods], rotation=25, ha="right")
    ax.set_ylabel("Iterations"); ax.set_title(f"Convergence Speed — {domain.title()}")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    p = out / f"convergence_{domain}.pdf"; fig.savefig(p); plt.close(); print(f"  {p}")


# ═══════════════════════════════════════════════
# TABLE: Summary (CSV + LaTeX)
# ═══════════════════════════════════════════════

def gen_table(df, out):
    if df.empty: return
    avg_all = df.groupby(["method","domain"])["avg_ctr"].mean().reset_index().rename(columns={"avg_ctr":"avg"})
    mx = df.groupby(["method","domain","topic"])["iteration"].max().reset_index()
    last = df.merge(mx, on=["method","domain","topic","iteration"])
    avg_last = last.groupby(["method","domain"])["avg_ctr"].mean().reset_index().rename(columns={"avg_ctr":"last"})
    s = avg_all.merge(avg_last, on=["method","domain"])
    s["avg"] = (s["avg"]*100).round(1); s["last"] = (s["last"]*100).round(1)
    s["method"] = s["method"].map(lambda m: LABELS.get(m,m))
    s.to_csv(out / "summary_table.csv", index=False)
    with open(out / "summary_table.tex", "w") as f:
        f.write(s.to_latex(index=False, float_format="%.1f"))
    print(f"  {out / 'summary_table.csv'}")


# ═══════════════════════════════════════════════
# Master
# ═══════════════════════════════════════════════

ALL_PLOTS = {
    "ctr_curves": lambda df,b,d,o: plot_ctr_curves(df,d,o),
    "qs_curves": lambda df,b,d,o: plot_qs_curves(df,b,d,o),
    "per_persona": lambda df,b,d,o: plot_per_persona(df,b,d,o),
    "model_cmp": lambda df,b,d,o: plot_model_cmp(df,d,o),
    "eval_cmp": lambda df,b,d,o: plot_eval_cmp(df,d,o),
    "regret": lambda df,b,d,o: plot_regret(df,d,o),
    "scores": lambda df,b,d,o: plot_scores(b,d,o),
    "heatmap": lambda df,b,d,o: plot_heatmap(df,d,o),
    "convergence": lambda df,b,d,o: plot_convergence(df,d,o),
}

def generate_all(base, out):
    out.mkdir(parents=True, exist_ok=True); setup()
    df = load_all_csv(base)
    if df.empty: print("No data found."); return
    print(f"Loaded {len(df)} rows | methods={df['method'].unique().tolist()} | domains={df['domain'].unique().tolist()}")

    for domain in df["domain"].unique():
        print(f"\n--- {domain} ---")
        for name, fn in ALL_PLOTS.items():
            print(f"  [{name}]"); fn(df, base, domain, out)

    print("\n  [cost]"); plot_cost(df, out)
    print("  [table]"); gen_table(df, out)
    print(f"\nAll plots → {out}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="./results")
    p.add_argument("--output-dir", default="./figures")
    p.add_argument("--plot", default="all", choices=list(ALL_PLOTS.keys()) + ["all","cost","table"])
    p.add_argument("--domain", default=None)
    args = p.parse_args()

    base, out = Path(args.results_dir), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.plot == "all":
        generate_all(base, out)
    else:
        setup(); df = load_all_csv(base)
        d = args.domain or (df["domain"].iloc[0] if not df.empty else "wikipedia")
        if args.plot == "cost": plot_cost(df, out)
        elif args.plot == "table": gen_table(df, out)
        else: ALL_PLOTS[args.plot](df, base, d, out)

if __name__ == "__main__":
    main()