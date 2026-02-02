from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd


@dataclass
class TruthfulQASample:
    id: int
    q_type: str
    category: str
    question: str
    best_answer: str
    incorrect_answers: List[str]
    source: Optional[str] = None


def load_truthfulqa_csv(path: str | Path) -> list[TruthfulQASample]:
    path = Path(path)
    df = pd.read_csv(path)

    required = {"Type", "Category", "Question", "Best Answer", "Incorrect Answers"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"TruthfulQA.csv is missing columns: {sorted(missing)}")

    samples: list[TruthfulQASample] = []
    for idx, row in df.iterrows():
        incorrect_raw = str(row.get("Incorrect Answers", "")).strip()
        incorrect_list = [x.strip() for x in incorrect_raw.split(";") if x.strip()] if incorrect_raw else []

        samples.append(
            TruthfulQASample(
                id=int(idx),
                q_type=str(row.get("Type", "")).strip(),
                category=str(row.get("Category", "")).strip(),
                question=str(row.get("Question", "")).strip(),
                best_answer=str(row.get("Best Answer", "")).strip(),
                incorrect_answers=incorrect_list,
                source=str(row.get("Source", "")).strip() if "Source" in df.columns else None,
            )
        )
    return samples


def make_subset(
    samples: list[TruthfulQASample],
    q_type: str = "Adversarial",
    n: int = 50,
    seed: int = 42,
) -> list[TruthfulQASample]:
    """
    Deterministic sampling:
    - Filter by Type
    - Shuffle with seed
    - Take first n
    """
    filtered = [s for s in samples if s.q_type == q_type and s.question]
    if not filtered:
        raise ValueError(f"No samples found for Type='{q_type}'")

    # deterministic shuffle without numpy:
    rng = seed
    idxs = list(range(len(filtered)))
    for i in range(len(idxs) - 1, 0, -1):
        rng = (1103515245 * rng + 12345) & 0x7FFFFFFF
        j = rng % (i + 1)
        idxs[i], idxs[j] = idxs[j], idxs[i]

    subset = [filtered[i] for i in idxs[:n]]
    return subset


def save_subset_json(subset: list[TruthfulQASample], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = [
        {
            "id": s.id,
            "type": s.q_type,
            "category": s.category,
            "question": s.question,
            "best_answer": s.best_answer,
            "incorrect_answers": s.incorrect_answers,
            "source": s.source,
        }
        for s in subset
    ]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
