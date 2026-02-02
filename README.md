# Truth, Lies, and Reasoning Machines

This repository contains the code and results for a course project investigating how large language models (LLMs) reason under false, noisy, and adversarial information.

The project evaluates whether LLMs accept, reject, or correct misleading claims, and how explicit self-verification prompts affect their factual accuracy and reasoning stability.

---

## Project Overview

We study LLM behavior under three controlled conditions:

- **Baseline**: Question only, no misleading context  
- **Noisy**: Question with an explicitly unreliable false claim  
- **Adversarial**: Question with a confidently stated but incorrect claim framed as authoritative  

For each condition, the model produces:
1. an initial answer  
2. a verified answer after a self-check prompt  

The experiment measures factual accuracy, self-correction, and regression across conditions.

---

## Models Evaluated

- **Claude (Anthropic API)**  
- **Qwen-2.5-7B-Instruct (Hugging Face Inference API)**  

Both models are evaluated using the same prompts, dataset subset, and evaluation pipeline.

---

## Dataset

- TruthfulQA  
- A fixed subset of 50 questions is used for all experiments  

Files:
- `data/TruthfulQA.csv`
- `data/truthfulqa_subset_50.json`

---

## Repository Structure

```
.
├── data/                  # Dataset and subset
├── src/                   # Core logic (prompts, models, evaluation)
├── results/               # Outputs, annotations, plots, metrics
├── run_experiment.py      # Run experiments (Claude)
├── run_hf.py              # Run experiments (Hugging Face model)
├── compute_metrics.py     # Compute accuracy and consistency metrics
├── plot_results.py        # Generate accuracy plots
└── README.md
```

---

## How to Run

### 1. Setup environment
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Configure API keys
Create a `.env` file with the required API keys (not included in the repo).

### 3. Run experiments
```bash
python run_experiment.py   # Claude
python run_hf.py           # Hugging Face model
```

### 4. Evaluate results
```bash
python compute_metrics.py
python plot_results.py
```

---

## Evaluation

Outputs are evaluated using:
- factual accuracy (before and after self-verification)
- self-correction rate
- regression rate (logical inconsistency after verification)
- qualitative manual inspection of reasoning behavior

Annotation files are included in `results/`.

---
