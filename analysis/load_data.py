"""
load_data.py
------------
Shared helper imported by all three analysis scripts.
Loads both CSVs, merges them, and adds derived columns.

Usage (from the other scripts):
    from load_data import load_df, CONDITIONS, COND_LABELS
"""

import os
import pandas as pd

# ── File paths ────────────────────────────────────────────────────────────────
CLAUDE_FILE = "../results/annotation_table_claude.csv"
QWEN_FILE   = "../results/annotation_table_qwen.csv"

# ── Constants used across all scripts ────────────────────────────────────────
CONDITIONS  = ["baseline", "noisy", "adversarial"]
COND_LABELS = ["Baseline", "Noisy", "Adversarial"]

# ── Shared plot colors ────────────────────────────────────────────────────────
COLOR_CLAUDE = "steelblue"
COLOR_QWEN   = "darkorange"
COLOR_GREEN  = "seagreen"
COLOR_RED    = "firebrick"


def load_df():
    """Load and merge both annotation CSVs into a single DataFrame."""
    claude_df = pd.read_csv(CLAUDE_FILE)
    qwen_df   = pd.read_csv(QWEN_FILE)

    claude_df["model_name"] = "Claude"
    qwen_df["model_name"]   = "Qwen"

    df = pd.concat([claude_df, qwen_df], ignore_index=True)

    # Make sure flag columns are integers
    for col in ["correct_first", "correct_verified", "hallucination_first"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Derived columns
    df["self_corrected"] = ((df["correct_first"] == 0) & (df["correct_verified"] == 1)).astype(int)
    df["regressed"]      = ((df["correct_first"] == 1) & (df["correct_verified"] == 0)).astype(int)

    print(f"Loaded {len(df)} rows | models: {df['model_name'].unique().tolist()} "
          f"| conditions: {df['condition'].unique().tolist()}")
    return df