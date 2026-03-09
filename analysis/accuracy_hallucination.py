"""
accuracy_hallucination.py
--------------------------------
Analyzes model accuracy and hallucination rates across conditions.

Produces:
  plots/1a_accuracy_first.png      — First-answer accuracy per condition
  plots/1b_accuracy_verified.png   — Post-verification accuracy per condition
  plots/1c_hallucination.png       — Hallucination rate per condition

Run:
  python accuracy_hallucination.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_df, CONDITIONS, COND_LABELS, COLOR_CLAUDE, COLOR_QWEN

os.makedirs("plots", exist_ok=True)

MODELS = ["Claude", "Qwen"]
COLORS = [COLOR_CLAUDE, COLOR_QWEN]


def pct(series):
    return series.mean() * 100


def print_summary_table(df):
    print("\nAccuracy & Hallucination Summary")
    print("-" * 60)
    print(f"{'Model':<8} {'Condition':<12} {'1st Acc':>8} {'Veri Acc':>9} {'Halluc':>8}")
    print("-" * 60)
    for model in MODELS:
        for cond in CONDITIONS:
            sub = df[(df["model_name"] == model) & (df["condition"] == cond)]
            f = pct(sub["correct_first"])
            v = pct(sub["correct_verified"])
            h = pct(sub["hallucination_first"])
            print(f"{model:<8} {cond:<12} {f:>7.1f}% {v:>8.1f}% {h:>7.1f}%")
        print()


def plot_accuracy(df, metric, filename, title):
    """Grouped bar chart of accuracy per condition for both models."""
    # Build a (condition x model) table of percentages
    table = (
        df.groupby(["condition", "model_name"])[metric]
        .apply(pct)
        .unstack("model_name")
        .reindex(CONDITIONS)
    )

    x = np.arange(len(CONDITIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, table["Claude"], width, label="Claude", color=COLOR_CLAUDE)
    ax.bar(x + width / 2, table["Qwen"],   width, label="Qwen",   color=COLOR_QWEN)

    # Value labels on top of each bar
    for bars, offset in [(table["Claude"], -width / 2), (table["Qwen"], width / 2)]:
        for xi, val in zip(x, bars):
            ax.text(xi + offset, val + 0.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = f"plots/{filename}"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_hallucination(df):
    """Bar chart of hallucination rate per condition for both models."""
    table = (
        df.groupby(["condition", "model_name"])["hallucination_first"]
        .apply(pct)
        .unstack("model_name")
        .reindex(CONDITIONS)
    )

    x = np.arange(len(CONDITIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, table["Claude"], width, label="Claude", color=COLOR_CLAUDE)
    ax.bar(x + width / 2, table["Qwen"],   width, label="Qwen",   color=COLOR_QWEN)

    for bars, offset in [(table["Claude"], -width / 2), (table["Qwen"], width / 2)]:
        for xi, val in zip(x, bars):
            ax.text(xi + offset, val + 0.2, f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS)
    ax.set_ylim(0, 20)
    ax.set_ylabel("Hallucination Rate (%)")
    ax.set_title("Hallucination Rate by Condition")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = "results/plots/1c_hallucination.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


if __name__ == "__main__":
    df = load_df()
    print_summary_table(df)

    plot_accuracy(df, "correct_first",    "1a_accuracy_first.png",    "First-Answer Accuracy by Condition")
    plot_accuracy(df, "correct_verified", "1b_accuracy_verified.png", "Post-Verification Accuracy by Condition")
    plot_hallucination(df)

    print("\nDone. See plots/1a, 1b, 1c.")