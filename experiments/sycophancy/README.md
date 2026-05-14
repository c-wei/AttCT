# Sycophancy

The headline threat model — does consistency training reduce the model's tendency to flip its MCQ answer under biased hints?

- **`eval_sycophancy_behavioral.py`** — post-hoc MCQ sycophancy on the BCT held-out set. F1 + `not_sycophantic_rate`. Called by [`run_evals.py`](../../run_evals.py).
- **`evaluate_sycophancy.py`** — in-training logprob `SycophancyEvaluator`, instantiated lazily by `run.py`.
- **`evaluate_bct.py`** — BRR (Biased Response Rate) from cot-transparency test sets. Both standalone (`run_brr_eval`) and shared-vLLM (`run_brr_with_llm`) entries.

Training data lives in [`../../datasets/sycophancy_bct/`](../../datasets/) (canonical) or `datasets/fresh_bct_<model>/` (per-model regenerations). Wrapping templates: 12 sycophancy patterns in [`../../data/wrappers.py`](../../data/wrappers.py).
