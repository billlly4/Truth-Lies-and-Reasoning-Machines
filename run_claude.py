from src.model_interface import ModelConfig, ClaudeChatModel
from src.runner import run

model = ClaudeChatModel(ModelConfig(name="claude-haiku-4-5-20251001"))

run(
    model=model,
    data_csv="data/TruthfulQA.csv",
    out_jsonl="results/raw_outputs.jsonl",
    n=50,      
    seed=42,
    q_type="Adversarial",
)

print("Done. Wrote results/raw_outputs.jsonl")
