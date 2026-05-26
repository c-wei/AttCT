# `datasets/`

On-disk training data. Python loaders are in [`../data/`](../data/).

**Sycophancy training**
- `sycophancy_bct/` — canonical BCT-style training set from Chua et al. 2024. Used directly by AttCT / ACT / MLPCT / BCT.
- `fresh_bct_<model>/` — per-model regenerated BCT pairs. Six directories present: `gemma3_4b`, `gemma3_27b`, `llama31_8b`, `qwen3_4b`, `qwen3_8b` (the 5-model headline set) plus `gemma2_9b` (earlier exploration, not in the headline). Schema: `control_{cot,non_cot}.jsonl`, `bct_{cot,non_cot}.jsonl`, plus 4000/1000 `_train` / `_eval` splits (seed 0, via `scripts/split_bct_train_eval.py`). Loaders auto-prefer the splits when present.

**Prefill (paper §4.2)**
- `attacks.csv` — 101 prefill prefix strings (`"Sure! "`, `"Of course, here's how:"`, …) loaded by `data/prefill_dataset.py`.
- `harmful_behaviors_pair.csv` — ClearHarm prompts paired with refusal targets.
- `clearharm_prefills.csv` — ClearHarm in the paper's per-strategy prefill format (subset of the per-strategy ClearHarm prefills dataset released on de-anonymization; 23 prefills per harmful prompt across the Struppek et al. 2025 strategy taxonomy).
- `advbench_prefills.csv` — 50 OOD prompts from AdvBench, used as the held-out PAR eval set.

**Frustration (paper §4.3)**
- `wildchat_frustration_train.jsonl` (25 prompts) — Gemini-filtered WildChat sample. Paper §4.3 + Appendix D describe a 50-prompt training corpus that produces 1,868 BCT-frustration pairs by selecting $(c_t, y_t)$ pairs with judge score ≥5 across rejection trajectories; the on-disk file is a 25-prompt subset.
- `wildchat_frustration_train_subset.jsonl` (20 prompts) — smaller subset for ablations.
- `wildchat_frustration_v3_test.jsonl` (25 prompts) — held-out evaluation set (matches paper "25-prompt held-out").
- `math_puzzles_train.jsonl` (15 puzzles) — lateral-thinking trick questions for the math-frustration training pool.
- `math_puzzles_v3_test.jsonl` (15 puzzles) — held-out math eval. Paper splits 30 puzzles 15/15 train/eval.

**Adding a new model to the sycophancy table:**
1. `python scripts/generate_fresh_bct_data.py --model <hf-repo> --output datasets/fresh_bct_<model>/`
2. `python scripts/split_bct_train_eval.py datasets/fresh_bct_<model>/`
3. Add `configs/bct_lora_<model>.yaml`, `configs/act_sycophancy_<model>_v2.yaml`, `configs/experiment_mlp_<model>.yaml`.

JSONL convention: one record per line, `{"prompt": ..., "response": ..., ...}`. Splits are seed-0 deterministic.
