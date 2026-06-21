"""Generate baseline and bandit benchmark summaries and figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS = Path("results")
OUTPUT = RESULTS / "bandit_10x10"
DOMAINS = ["wikipedia", "ecommerce"]
BASELINE_METHODS = [
    "no_drop",
    "random_ctr",
    "partial_ctr",
    "full_ctr",
    "explore_exploit",
]
BANDIT_METHODS = ["bandit_ucb", "bandit_epsilon", "bandit_thompson"]
COMPARISON_METHODS = ["explore_exploit", *BANDIT_METHODS]

LABELS = {
    "no_drop": "No drop",
    "random_ctr": "Random CTR",
    "partial_ctr": "Partial CTR",
    "full_ctr": "Full CTR",
    "explore_exploit": "Fixed explore-exploit",
    "bandit_ucb": "UCB (c=0.002)",
    "bandit_epsilon": "Epsilon-greedy",
    "bandit_thompson": "Thompson sampling",
}
COLORS = {
    "no_drop": "#6B7280",
    "random_ctr": "#2563EB",
    "partial_ctr": "#D97706",
    "full_ctr": "#059669",
    "explore_exploit": "#7C3AED",
    "bandit_ucb": "#DC2626",
    "bandit_epsilon": "#0891B2",
    "bandit_thompson": "#DB2777",
}
ARMS = ["explore", "exploit", "both"]
ARM_COLORS = {"explore": "#2563EB", "exploit": "#D97706", "both": "#059669"}


def run_dir(domain, method):
    suffix = "__ucb_c002" if method == "bandit_ucb" else ""
    return RESULTS / (
        f"{domain}__{method}__gpt-4o-mini__"
        f"film_best_dimensions_mpnet__user{suffix}"
    )


def load_methods(methods):
    frames = []
    for domain in DOMAINS:
        for method in methods:
            path = run_dir(domain, method) / "results.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Missing benchmark result: {path}")
            frame = pd.read_csv(path)
            frame = frame[
                (frame["generator_model"] == "gpt-4o-mini")
                & (frame["evaluator"] == "film_best_dimensions_mpnet")
                & (frame["user_level"] == True)  # noqa: E712
            ].copy()
            if len(frame) != 150 or frame["topic"].nunique() != 10:
                raise ValueError(
                    f"Incomplete benchmark result: {path} "
                    f"({len(frame)} rows, {frame['topic'].nunique()} topics)"
                )
            if frame.duplicated(["topic", "iteration"]).any():
                raise ValueError(f"Duplicate topic iterations in {path}")
            frame["method"] = method
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def add_regret(frame):
    oracle = (
        frame.groupby(["domain", "topic", "iteration"], as_index=False)["avg_ctr"]
        .max()
        .rename(columns={"avg_ctr": "oracle_ctr"})
    )
    result = frame.merge(oracle, on=["domain", "topic", "iteration"])
    result["regret"] = result["oracle_ctr"] - result["avg_ctr"]
    return result


def summarize(frame):
    rows = []
    for (domain, method), group in frame.groupby(["domain", "method"]):
        trajectory = group.groupby("iteration")["avg_ctr"].mean().sort_index()
        topic_metrics = []
        for _, topic in group.groupby("topic"):
            values = topic.sort_values("iteration")["avg_ctr"].to_numpy()
            topic_metrics.append(
                {
                    "initial_ctr": values[0],
                    "final_ctr": values[-1],
                    "auc_mean_ctr": values.mean(),
                    "peak_ctr": values.max(),
                    "final_improvement": values[-1] - values[0],
                    "peak_improvement": values.max() - values[0],
                }
            )
        metrics = pd.DataFrame(topic_metrics).mean().to_dict()
        regret = (
            group.groupby("iteration")["regret"].mean().sort_index().sum()
            if "regret" in group
            else np.nan
        )
        rows.append(
            {
                "domain": domain,
                "method": method,
                "n_topics": group["topic"].nunique(),
                "iterations": group["iteration"].nunique(),
                **metrics,
                "trajectory_peak_ctr": trajectory.max(),
                "cumulative_regret": regret,
            }
        )
    return pd.DataFrame(rows).sort_values(["domain", "method"]).reset_index(drop=True)


def style():
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def save(fig, name):
    path = OUTPUT / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def plot_trajectory(frame, domain, methods, name, title):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    subset = frame[frame["domain"] == domain]
    for method in methods:
        method_rows = subset[subset["method"] == method]
        stats = method_rows.groupby("iteration")["avg_ctr"].agg(["mean", "sem"])
        x = stats.index.to_numpy()
        mean = stats["mean"].to_numpy() * 100
        sem = stats["sem"].fillna(0).to_numpy() * 100
        ax.plot(
            x,
            mean,
            marker="o",
            markersize=3.5,
            linewidth=2,
            color=COLORS[method],
            label=LABELS[method],
        )
        ax.fill_between(x, mean - sem, mean + sem, color=COLORS[method], alpha=0.12)
    ax.set(title=title, xlabel="Iteration", ylabel="Average CTR (%)")
    ax.set_xticks(range(15))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=9)
    save(fig, name)


def plot_metric(summary, column, ylabel, name):
    methods = COMPARISON_METHODS
    domains = DOMAINS
    x = np.arange(len(methods))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for index, domain in enumerate(domains):
        values = [
            summary.loc[
                (summary["domain"] == domain) & (summary["method"] == method), column
            ].iloc[0]
            * 100
            for method in methods
        ]
        offset = (index - 0.5) * width
        bars = ax.bar(
            x + offset,
            values,
            width,
            label=domain.title(),
            color=["#2563EB", "#D97706"][index],
            alpha=0.88,
        )
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x, [LABELS[method] for method in methods], rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    save(fig, name)


def plot_arm_distribution(bandits):
    counts = (
        bandits.groupby(["domain", "method", "chosen_arm"])
        .size()
        .rename("count")
        .reset_index()
    )
    counts["share"] = counts["count"] / counts.groupby(
        ["domain", "method"]
    )["count"].transform("sum")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        subset = counts[counts["domain"] == domain]
        bottoms = np.zeros(len(BANDIT_METHODS))
        for arm in ARMS:
            values = [
                subset.loc[
                    (subset["method"] == method) & (subset["chosen_arm"] == arm),
                    "share",
                ].sum()
                for method in BANDIT_METHODS
            ]
            ax.bar(
                range(len(BANDIT_METHODS)),
                values,
                bottom=bottoms,
                color=ARM_COLORS[arm],
                label=arm.title(),
            )
            bottoms += np.array(values)
        ax.set_title(domain.title())
        ax.set_xticks(
            range(len(BANDIT_METHODS)),
            [LABELS[method] for method in BANDIT_METHODS],
            rotation=17,
            ha="right",
        )
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Arm selection share")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Arm",
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.subplots_adjust(top=0.86)
    save(fig, "arm_selection_distribution.png")


def plot_regret(comparison):
    regret = add_regret(comparison)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, domain in zip(axes, DOMAINS):
        subset = regret[regret["domain"] == domain]
        for method in COMPARISON_METHODS:
            curve = (
                subset[subset["method"] == method]
                .groupby("iteration")["regret"]
                .mean()
                .sort_index()
                .cumsum()
                * 100
            )
            ax.plot(
                curve.index,
                curve.values,
                linewidth=2,
                color=COLORS[method],
                label=LABELS[method],
            )
        ax.set_title(domain.title())
        ax.set_xlabel("Iteration")
        ax.set_xticks(range(15))
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Cumulative regret (CTR percentage points)")
    axes[1].legend(fontsize=8)
    save(fig, "cumulative_regret.png")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()

    baseline = load_methods(BASELINE_METHODS)
    comparison = load_methods(COMPARISON_METHODS)
    bandits = comparison[comparison["method"].isin(BANDIT_METHODS)].copy()

    baseline_with_regret = add_regret(baseline)
    comparison_with_regret = add_regret(comparison)
    baseline_summary = summarize(baseline_with_regret)
    comparison_summary = summarize(comparison_with_regret)
    bandit_summary = comparison_summary[
        comparison_summary["method"].isin(BANDIT_METHODS)
    ].reset_index(drop=True)

    baseline_summary.to_csv(OUTPUT / "baseline_summary.csv", index=False)
    bandit_summary.to_csv(OUTPUT / "bandit_summary.csv", index=False)
    comparison_summary.to_csv(
        OUTPUT / "method_comparison_summary.csv", index=False
    )

    for domain in DOMAINS:
        plot_trajectory(
            baseline,
            domain,
            BASELINE_METHODS,
            f"ctr_trajectory_baseline_{domain}.png",
            f"Baseline CTR trajectory: {domain.title()}",
        )
        plot_trajectory(
            comparison,
            domain,
            COMPARISON_METHODS,
            f"bandit_vs_fixed_{domain}.png",
            f"Bandits vs fixed explore-exploit: {domain.title()}",
        )

    plot_metric(
        comparison_summary,
        "final_ctr",
        "Final average CTR (%)",
        "final_ctr_comparison.png",
    )
    plot_metric(
        comparison_summary,
        "auc_mean_ctr",
        "Mean CTR / normalized AUC (%)",
        "auc_comparison.png",
    )
    plot_arm_distribution(bandits)
    plot_regret(comparison)

    print(OUTPUT / "baseline_summary.csv")
    print(OUTPUT / "bandit_summary.csv")
    print(OUTPUT / "method_comparison_summary.csv")


if __name__ == "__main__":
    main()
