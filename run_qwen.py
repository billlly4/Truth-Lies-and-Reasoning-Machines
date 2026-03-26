from dotenv import load_dotenv
load_dotenv()

from src.runner import run

from qwen_interface import HFConfig, HFChatModel
model = HFChatModel(HFConfig(name="Qwen/Qwen2.5-7B-Instruct"))

run(
    model=model,
    data_csv="data/TruthfulQA.csv",
    out_jsonl="results/raw_outputs_hf.jsonl",
    n=50,
    seed=42,
)
print("Done. Wrote results/raw_outputs_hf.jsonl")
