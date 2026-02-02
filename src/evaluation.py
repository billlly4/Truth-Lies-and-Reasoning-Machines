from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)
    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    preferred = [
        "question_id",
        "type",
        "category",
        "model",
        "condition",
        "question",
        "best_answer",
        "incorrect_fact_used",
        "first_answer",
        "verified_answer",
        "incorrect_answers",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]
    return df[cols]


def normalize_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def quick_auto_label(best: str, answer: str) -> str:
    b = normalize_text(best)
    a = normalize_text(answer)
    if b and b in a:
        return "1"
    return ""


def build_annotation_csv(
    jsonl_path: str | Path,
    out_csv_path: str | Path,
    auto_prefill: bool = True,
) -> pd.DataFrame:
    records = load_jsonl(jsonl_path)
    df = to_dataframe(records)

    df["correct_first"] = ""
    df["correct_verified"] = ""
    df["hallucination_first"] = ""
    df["notes"] = ""

    if auto_prefill:
        df["correct_first"] = [
            quick_auto_label(b, a) for b, a in zip(df["best_answer"], df["first_answer"])
        ]
        df["correct_verified"] = [
            quick_auto_label(b, a) for b, a in zip(df["best_answer"], df["verified_answer"])
        ]

    out_csv_path = Path(out_csv_path)
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return df


def _safe_mean01(series: pd.Series) -> float | None:
    """
    Accepts 0/1 as ints, floats, or strings. Ignores blanks/NaN.
    Returns None if no valid values.
    """
    s = pd.to_numeric(series, errors="coerce")  # converts "1", 1.0, etc. to numeric; blanks -> NaN
    s = s[s.isin([0, 1])]
    if len(s) == 0:
        return None
    return float(s.mean())


def compute_metrics(annotated_csv_path: str | Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    df = pd.read_csv(annotated_csv_path)

    # Condition-level accuracies
    rows = []
    for cond in sorted(df["condition"].unique()):
        sub = df[df["condition"] == cond]
        rows.append(
            {
                "condition": cond,
                "n": len(sub),
                "acc_first": _safe_mean01(sub["correct_first"]) if "correct_first" in sub.columns else None,
                "acc_verified": _safe_mean01(sub["correct_verified"]) if "correct_verified" in sub.columns else None,
            }
        )
    summary = pd.DataFrame(rows)

    # Self-correction/regression
    cf = pd.to_numeric(df.get("correct_first"), errors="coerce")
    cv = pd.to_numeric(df.get("correct_verified"), errors="coerce")

    wrong_first = df[cf == 0]
    if len(wrong_first) > 0:
        corrected = wrong_first[pd.to_numeric(wrong_first["correct_verified"], errors="coerce") == 1]
        self_correction_rate = len(corrected) / len(wrong_first)
    else:
        self_correction_rate = None

    correct_first = df[cf == 1]
    if len(correct_first) > 0:
        regressed = correct_first[pd.to_numeric(correct_first["correct_verified"], errors="coerce") == 0]
        regression_rate = len(regressed) / len(correct_first)
    else:
        regression_rate = None

    meta = {
        "self_correction_rate": self_correction_rate,
        "regression_rate": regression_rate,
        "total_rows": len(df),
    }
    return summary, meta
