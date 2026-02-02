from src.evaluation import compute_metrics

#summary, meta = compute_metrics("results/annotation_table.csv")
summary, meta = compute_metrics("results/annotation_table_hf.csv")

print("\n=== Condition Summary ===")
print(summary.to_string(index=False))

print("\n=== Meta Metrics ===")
for k, v in meta.items():
    print(f"{k}: {v}")
