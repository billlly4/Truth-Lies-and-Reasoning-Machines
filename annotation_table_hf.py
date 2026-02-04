from src.evaluation import build_annotation_csv

df = build_annotation_csv(
    jsonl_path="results/raw_outputs_hf.jsonl",
    out_csv_path="results/annotation_table_hf.csv",
    auto_prefill=True,  
)

print("Wrote results/annotation_table_hf.csv")
print("Rows:", len(df))
print(df[["condition", "question_id", "correct_first", "correct_verified"]].head(10))
