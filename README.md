
# Truth, Lies, and Reasoning Machines

This repository contains the code and results for a project investigating how large language models (LLMs) behave when reasoning under **misinformation and misleading context**.

The project evaluates whether models:
- accept false information,
- resist misleading claims,
- or correct themselves after explicit verification prompts.

Two models are evaluated under controlled conditions using the **TruthfulQA dataset**.

---

# Research Goal

Large Language Models can produce fluent but factually incorrect answers when exposed to misleading or adversarial information. This project studies:

1. **How misinformation affects reasoning**
2. **Whether models can detect and reject false claims**
3. **Whether self‑verification improves reliability**
4. **Which knowledge domains are most vulnerable to hallucinations**

---

# Experimental Design

Each question is tested under **three prompting conditions**:

| Condition | Description |
|--------|-------------|
| Baseline | Question only |
| Noisy | Question with a misleading statement that may be false |
| Adversarial | Question with a confidently stated incorrect claim framed as authority |

For every condition the model produces:

1. **First answer**
2. **Verified answer** after a self‑check prompt

This allows measurement of reasoning stability and self‑correction behavior.

---

# Models Evaluated

The same experiment pipeline is used for both models:

- **Claude (Anthropic API)**
- **Qwen‑2.5‑7B‑Instruct (HuggingFace Inference API)**

---

# Dataset

The experiments use **TruthfulQA**, a dataset designed to evaluate whether models produce truthful answers or mimic common misconceptions.

A **fixed subset of 50 adversarial questions** is used to keep experiments reproducible.

Files:

```
data/
    TruthfulQA.csv
    truthfulqa_subset_50.json
```

---

# Repository Structure

```
.
├── analysis/                # All evaluation and analysis scripts
│   ├── accuracy_hallucination.py
│   ├── verification.py
│   ├── categories.py
│   └── load_data.py
│
├── data/                    # Dataset files
│   ├── TruthfulQA.csv
│   └── truthfulqa_subset_50.json
│
├── results/
│   ├── raw_outputs_claude.jsonl
│   ├── raw_outputs_qwen.jsonl
│   ├── annotation_table_claude.csv
│   ├── annotation_table_qwen.csv
│   └── plots/
│        ├── 1c_hallucination.png
│        ├── 2a_correction_regression.png
│        ├── 2b_accuracy_delta.png
│        ├── 3a_category_accuracy.png
│        └── 3b_category_heatmap.png
│
├── src/                     # Core experiment pipeline
│   ├── runner.py
│   ├── prompts.py
│   ├── data_loader.py
│   ├── claude_interface.py
│   └── qwen_interface.py
│
├── run_claude.py            # Run experiment using Claude
├── run_qwen.py              # Run experiment using Qwen
├── requirements.txt
└── README.md
```

---

# Pipeline Overview

The project follows a reproducible experimental pipeline:

```
Dataset
   ↓
Prompt generation
   ↓
Model inference
   ↓
Raw outputs (JSONL)
   ↓
Manual annotation
   ↓
Analysis scripts
   ↓
Plots and metrics
```

---

# Running the Project

## 1. Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows:

```bash
.venv\Scripts\activate
```

---

# 2. Add API keys

Create a `.env` file in the root directory:

```
ANTHROPIC_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

---

# 3. Run experiments

Generate raw model outputs:

```
python run_claude.py
python run_qwen.py
```

Outputs will be saved to:

```
results/raw_outputs_claude.jsonl
results/raw_outputs_qwen.jsonl
```

---

# 4. Annotate results

Model answers are manually labeled for:

- correctness of first answer
- correctness after verification
- hallucination presence

These annotations are stored in:

```
results/annotation_table_claude.csv
results/annotation_table_qwen.csv
```

---

# 5. Run analysis

## Accuracy and hallucination analysis

```
python analysis/accuracy_hallucination.py
```

Produces:

```
results/plots/1c_hallucination.png
```

---

## Verification analysis

```
python analysis/verification.py
```

Produces:

```
results/plots/2a_correction_regression.png
results/plots/2b_accuracy_delta.png
```

---

## Category analysis

```
python analysis/categories.py
```

Produces:

```
results/plots/3a_category_accuracy.png
results/plots/3b_category_heatmap.png
```

---

# Evaluation Metrics

The project measures:

### Accuracy
- correctness of first response
- correctness after verification

### Hallucination Rate
Percentage of answers containing unsupported factual claims.

### Self‑Correction
Cases where verification changed a wrong answer into a correct one.

### Regression
Cases where verification changed a correct answer into a wrong one.

### Category Performance
Accuracy differences across knowledge domains.

---

# Key Research Questions

The experiments aim to answer:

- Do LLMs accept false claims when they appear authoritative?
- Does explicit self‑verification improve factual accuracy?
- Do models hallucinate more under adversarial context?
- Which categories of knowledge are most difficult?

---

# Reproducibility

All experiments use:

- fixed dataset subset
- deterministic sampling
- identical prompt templates across models

This ensures direct comparison between models.

---

# License

This repository is intended for research and educational purposes.
