# Sycophancy (paper §3.1)

The headline threat model — does training on BCT-style consistency reduce the model's tendency to flip its MCQ answer when given a biased hint (suggested answer, "are you sure?", post-hoc, etc.)?

| File | Purpose | Metric |
|---|---|---|
| `eval_sycophancy_behavioral.py` | MCQ sycophancy on the BCT held-out set. Resistance rate + F1. | F1, `not_sycophantic_rate` |
| `evaluate_sycophancy.py` | In-training logprob-based MMLU-sycophancy `SycophancyEvaluator`. Called from `run.py` during ACT/MLPCT training. | per-bias correct rate |
| `evaluate_bct.py` | BRR (Biased Response Rate) from cot-transparency test sets. `run_brr_eval(...)` and `run_brr_with_llm(...)` (shared-vLLM variant used by `run_evals.py`). | BRR Ratio |

Training data: `datasets/sycophancy_bct/` (canonical) or `datasets/fresh_bct_<model>/` (per-model regenerations for the 5 paper models). Wrapping templates: 12 sycophancy patterns in `data/wrappers.py`.
