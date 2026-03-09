from src.data_loader import load_truthfulqa_csv, make_subset, save_subset_json

samples = load_truthfulqa_csv("data/TruthfulQA.csv")
subset = make_subset(samples, q_type="Adversarial", n=50, seed=42)

print("Loaded:", len(samples))
print("Subset:", len(subset))
print("Example question:", subset[0].question)

save_subset_json(subset, "data/truthfulqa_subset_50.json")
print("Wrote: data/truthfulqa_subset_50.json")
