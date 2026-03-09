"""
categories.py
--------------------
Breaks down model performance by question category.
Shows which knowledge domains are hardest and where the
two models differ most.

Produces:
  results/plots/3a_category_accuracy.png   — Accuracy per category (both models, horizontal bars)
  results/plots/3b_category_heatmap.png    — Heatmap: category x condition accuracy per model

Run:
  python categories.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from load_data import load_df, CONDITIONS, COND_LABELS, COLOR_CLAUDE, COLOR_QWEN

os.makedirs("plots", exist_ok=True)

MODELS = ["Claude", "Qwen"]


def pct(series):
    return series.mean() * 100


def print_category_table(df):
    """Print a table of accuracy per category for each model."""
    cat_acc = (
        df.groupby(["model_name", "category"])["correct_first"]
        .apply(pct)
        .unstack("model_name")
        .round(1)
    )
    cat_acc["Gap (C-Q)"] = (cat_acc["Claude"] - cat_acc["Qwen"]).round(1)
    cat_acc = cat_acc.sort_values("Gap (C-Q)", ascending=False)

    print("\nFirst-Answer Accuracy by Category")
    print("-" * 50)
    print(f"{'Category':<30} {'Claude':>7} {'Qwen':>7} {'Gap':>7}")
    print("-" * 50)
    for cat, row in cat_acc.iterrows():
        print(f"{cat:<30} {row['Claude']:>6.1f}% {row['Qwen']:>6.1f}% {row['Gap (C-Q)']:>+6.1f}%")


def plot_category_accuracy(df):
    """Horizontal grouped bar: first-answer accuracy per category."""
    cat_acc = (
        df.groupby(["model_name", "category"])["correct_first"]
        .apply(pct)
        .unstack("model_name")
    )
    # Sort by average accuracy (hardest categories at top)
    cat_acc["avg"] = cat_acc.mean(axis=1)
    cat_acc = cat_acc.sort_values("avg").drop(columns="avg")

    categories = cat_acc.index.tolist()
    y = np.arange(len(categories))
    height = 0.35

    fig, ax = plt.subplots(figsize=(9, 10))
    ax.barh(y + height / 2, cat_acc["Claude"], height, label="Claude", color=COLOR_CLAUDE)
    ax.barh(y - height / 2, cat_acc["Qwen"],   height, label="Qwen",   color=COLOR_QWEN)

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlim(0, 110)
    ax.set_xlabel("First-Answer Accuracy (%)")
    ax.set_title("Accuracy by Category (sorted by difficulty)")
    ax.axvline(50, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    path = "results/plots/3a_category_accuracy.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_category_heatmap(df):
    """Two heatmaps side by side: category x condition for each model."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 9))

    # Determine shared category order (by average accuracy across both models)
    overall_order = (
        df.groupby("category")["correct_first"]
        .apply(pct)
        .sort_values()
        .index.tolist()
    )

    for ax, model in zip(axes, MODELS):
        sub = df[df["model_name"] == model]
        heat = (
            sub.groupby(["category", "condition"])["correct_first"]
            .apply(pct)
            .unstack("condition")
            .reindex(columns=CONDITIONS)
            .reindex(overall_order)
        )
        heat.columns = COND_LABELS

        sns.heatmap(
            heat, ax=ax,
            annot=True, fmt=".0f",
            cmap="RdYlGn",
            vmin=0, vmax=100,
            linewidths=0.4,
            cbar_kws={"label": "Accuracy (%)"},
            annot_kws={"size": 8},
        )
        ax.set_title(f"{model} — Accuracy by Category & Condition", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(axis="y", rotation=0, labelsize=8)
        ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    path = "results/plots/3b_category_heatmap.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


if __name__ == "__main__":
    df = load_df()
    print_category_table(df)

    plot_category_accuracy(df)
    plot_category_heatmap(df)

    print("\nDone. See plots/3a, 3b.")