"""
verification.py
----------------------
Analyzes the effect of self-verification prompts on model answers.
Looks at how often models self-correct or regress after being asked
to verify their first answer.

Produces:
  results/plots/2a_correction_regression.png  — Self-corrected vs regressed counts per model
  results/plots/2b_accuracy_delta.png         — Net accuracy change after verification per condition

Run:
  python verification.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_df, CONDITIONS, COND_LABELS, COLOR_CLAUDE, COLOR_QWEN, COLOR_GREEN, COLOR_RED

os.makedirs("plots", exist_ok=True)

MODELS = ["Claude", "Qwen"]


def pct(series):
    return series.mean() * 100


def print_verification_table(df):
    print("\nVerification Effect Summary")
    print("-" * 65)
    print(f"{'Model':<8} {'Condition':<12} {'1st Acc':>8} {'Veri Acc':>9} {'Delta':>7} {'Corr':>6} {'Regr':>6}")
    print("-" * 65)
    for model in MODELS:
        for cond in CONDITIONS:
            sub = df[(df["model_name"] == model) & (df["condition"] == cond)]
            f    = pct(sub["correct_first"])
            v    = pct(sub["correct_verified"])
            d    = v - f
            corr = sub["self_corrected"].sum()
            regr = sub["regressed"].sum()
            print(f"{model:<8} {cond:<12} {f:>7.1f}% {v:>8.1f}% {d:>+6.1f}% {corr:>6} {regr:>6}")
        print()


def plot_correction_regression(df):
    """Grouped bar: how many answers were self-corrected vs regressed, per model."""
    models = MODELS
    corrected = [df[df["model_name"] == m]["self_corrected"].sum() for m in models]
    regressed  = [df[df["model_name"] == m]["regressed"].sum()     for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 5))
    bars1 = ax.bar(x - width / 2, corrected, width, label="Self-corrected (wrong → right)", color=COLOR_GREEN)
    bars2 = ax.bar(x + width / 2, regressed,  width, label="Regressed (right → wrong)",     color=COLOR_RED)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                str(int(bar.get_height())), ha="center", va="bottom", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Number of cases")
    ax.set_title("Self-Correction vs Regression After Verification")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = "../results/plots/2a_correction_regression.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


def plot_accuracy_delta(df):
    """Bar chart of net accuracy change (verified - first) per condition per model."""
    x = np.arange(len(CONDITIONS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, (model, color) in enumerate(zip(MODELS, [COLOR_CLAUDE, COLOR_QWEN])):
        sub = df[df["model_name"] == model]
        deltas = [
            pct(sub[sub["condition"] == c]["correct_verified"]) -
            pct(sub[sub["condition"] == c]["correct_first"])
            for c in CONDITIONS
        ]
        offset = (i - 0.5) * width
        bar_colors = [COLOR_GREEN if d >= 0 else COLOR_RED for d in deltas]
        bars = ax.bar(x + offset, deltas, width, color=bar_colors,
                      edgecolor=color, linewidth=1.5, label=model)

        for xi, d in zip(x + offset, deltas):
            ax.text(xi, d + (0.1 if d >= 0 else -0.3),
                    f"{d:+.1f}%", ha="center", va="bottom", fontsize=9)

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(COND_LABELS)
    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_title("Net Accuracy Change After Verification\n(positive = verification helped)")
    ax.legend(title="Edge color = model")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    path = "../results/plots/2b_accuracy_delta.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"Saved → {path}")
    plt.close()


if __name__ == "__main__":
    df = load_df()
    print_verification_table(df)

    plot_correction_regression(df)
    plot_accuracy_delta(df)

    print("\nDone. See plots/2a, 2b.")