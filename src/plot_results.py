import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

RESULTS_DIR = "results"
PLOT_DIR = os.path.join(RESULTS_DIR, "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def plot_avg_ctr():
    df = pd.read_csv(f"{RESULTS_DIR}/ctr_results.csv", names=["iteration", "avg_ctr", "topic"])

    plt.figure(figsize=(8, 5))
    sns.lineplot(data=df, x="iteration", y="avg_ctr", hue="topic", marker="o")
    plt.title("Average CTR per Iteration")
    plt.ylabel("Average CTR")
    plt.xlabel("Iteration")
    plt.legend(title="Topic", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"{PLOT_DIR}/avg_ctr.png")
    plt.close()


def plot_boxplot_ctr():
    df = pd.read_csv(f"{RESULTS_DIR}/ctr_results.csv", names=["iteration", "avg_ctr", "topic"])
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="iteration", y="avg_ctr")
    plt.title("CTR Distribution per Iteration")
    plt.savefig(f"{PLOT_DIR}/boxplot_ctr.png")
    plt.close()


def plot_topic_comparison():
    df = pd.read_csv(f"{RESULTS_DIR}/ctr_results.csv", names=["iteration", "avg_ctr", "topic"])
    final = df[df["iteration"] == df["iteration"].max()]
    topic_means = final.groupby("topic")["avg_ctr"].mean().sort_values()

    plt.figure(figsize=(8, 5))
    sns.barplot(x=topic_means.values, y=topic_means.index, palette="viridis")
    plt.title("Final Average CTR per Topic")
    plt.xlabel("Avg CTR at Final Iteration")
    plt.savefig(f"{PLOT_DIR}/ctr_per_topic.png")
    plt.close()


def plot_all():
    plot_avg_ctr()
    plot_boxplot_ctr()
    plot_topic_comparison()
    print("✅ All plots saved in results/plots/")


if __name__ == "__main__":
    plot_all()
