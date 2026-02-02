from __future__ import annotations

import json
from pathlib import Path

from tqdm import tqdm

from .data_loader import load_truthfulqa_csv, make_subset
from .model_interface import BaseModel
from .prompts import SYSTEM_INSTRUCTION, all_conditions, build_self_check


def run(
    model: BaseModel,
    data_csv: str | Path,
    out_jsonl: str | Path,
    n: int = 50,
    seed: int = 42,
    q_type: str = "Adversarial",
) -> None:
    data_csv = Path(data_csv)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    samples = load_truthfulqa_csv(data_csv)
    subset = make_subset(samples, q_type=q_type, n=n, seed=seed)

    with out_jsonl.open("w", encoding="utf-8") as f:
        for sample in tqdm(subset, desc="Running questions"):
            for p in all_conditions(sample):
                first = model.generate(SYSTEM_INSTRUCTION, p.user_prompt)
                verify_prompt = build_self_check(first)
                verified = model.generate(SYSTEM_INSTRUCTION, verify_prompt)

                rec = {
                    "question_id": sample.id,
                    "type": sample.q_type,
                    "category": sample.category,
                    "question": sample.question,
                    "best_answer": sample.best_answer,
                    "incorrect_answers": sample.incorrect_answers,
                    "model": model.config.name,
                    "condition": p.condition,
                    "incorrect_fact_used": p.incorrect_fact_used,
                    "first_answer": first,
                    "verified_answer": verified,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
