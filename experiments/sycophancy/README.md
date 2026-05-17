# Sycophancy (paper §5.1)

Does the model flip its MCQ answer when given a biased hint (suggested answer, "are you sure?", post-hoc rationale, etc.)?

**Metric — BRR (Biased Reasoning Rate):**
$$\text{BRR} = P(\text{nudged} \mid \text{biased prompt}) - P(\text{nudged} \mid \text{clean prompt})$$
BRR Ratio = post-train BRR / base-model BRR (0 = full elimination; lower is better).

**Paper headline (5-model avg, Figure 1):** AttCT 0.019, MLPCT 0.19, BCT 0.78. Bias-on-MMLU: ACT 0.03, AttCT 0.085.

**Files**
- `evaluate_bct.py` — BRR via the cot-transparency test sets. Two entry points: `run_brr_eval()` (standalone, loads its own vLLM) and `run_brr_with_llm()` (shared-vLLM, called by [`../../run_evals.py`](../../run_evals.py)).
- `eval_sycophancy_behavioral.py` — post-hoc MCQ sycophancy on the BCT held-out set (4000/1000 split). Reports `not_sycophantic_rate`.
- `evaluate_sycophancy.py` — in-training logprob `SycophancyEvaluator`, instantiated lazily by `run.py` during the train loop.

Training data: [`../../datasets/sycophancy_bct/`](../../datasets/) (canonical, from Chua et al. 2024) or `datasets/fresh_bct_<model>/` (per-model regenerations for the 5 paper models). Wrappers applied on-the-fly via the 12 sycophancy templates in [`../../data/wrappers.py`](../../data/wrappers.py).
