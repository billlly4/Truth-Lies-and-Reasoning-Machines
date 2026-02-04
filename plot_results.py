import numpy as np
import matplotlib.pyplot as plt

from src.evaluation import compute_metrics

summary, meta = compute_metrics("results/annotation_table.csv")
#summary, meta = compute_metrics("results/annotation_table_hf.csv")

x = summary["condition"].tolist()

acc_first = np.array([np.nan if v is None else float(v) for v in summary["acc_first"]])
acc_verified = np.array([np.nan if v is None else float(v) for v in summary["acc_verified"]])

plt.figure()
plt.bar([i - 0.2 for i in range(len(x))], acc_first, width=0.4, label="First")
plt.bar([i + 0.2 for i in range(len(x))], acc_verified, width=0.4, label="Verified")
plt.xticks(range(len(x)), x)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Accuracy by Condition (Before vs After Self-Check)")
plt.legend()

out_path = "results/accuracy_by_condition.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print("Saved:", out_path)

print("\nSummary table:")
print(summary.to_string(index=False))
print("\nMeta:", meta)
print("\nNaN accuracies mean you haven't filled 0/1 for that condition yet.")
