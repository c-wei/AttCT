# `datasets/`

On-disk training data. Python loaders are in [`../data/`](../data/).

**Sycophancy training**
- `sycophancy_bct/` — canonical BCT-style training set from [Chua et al. 2024](https://arxiv.org/abs/2403.05518). Used directly by AttCT / ACT / MLPCT / BCT.
- `fresh_bct_<model>/` — per-model regenerated BCT pairs for each of the 5 paper models (`gemma2_9b`, `gemma3_4b`, `gemma3_27b`, `llama31_8b`, `qwen3_4b`, `qwen3_8b`). Schema: `control_{cot,non_cot}.jsonl`, `bct_{cot,non_cot}.jsonl`, plus 4000/1000 `_train` / `_eval` splits (seed 0, via `scripts/split_bct_train_eval.py`). Loaders auto-prefer the splits when present.

**Prefill (paper §3.4)**
- `attacks.csv` — 100 prefill prefix strings (`"Sure! "`, `"Of course, here's how:"`, …) loaded by `data/prefill_dataset.py`.
- `harmful_behaviors_pair.csv` — ClearHarm prompts paired with refusal targets.
- `clearharm_prefills.csv` — ClearHarm in the paper's per-strategy prefill format (subset of `carolinewei/ClearHarm_prefills` on HF Hub; 23 prefills per harmful prompt across the [Struppek et al. 2025](https://arxiv.org/) strategy taxonomy).
- `advbench_prefills.csv` — 50 OOD prompts from AdvBench, used as the held-out PAR eval set.

**Frustration (paper §3.5, §4.4)**
- `wildchat_frustration_train.jsonl` (50 prompts) — Gemini-filtered WildChat sample, used to build the 1,868 BCT-frustration training pairs.
- `wildchat_frustration_train_subset.jsonl` — smaller subset for ablations.
- `wildchat_frustration_v3_test.jsonl` (25 prompts) — held-out evaluation set.
- `math_puzzles_train.jsonl` (15 puzzles) — lateral-thinking trick questions for the math-frustration training pool.
- `math_puzzles_v3_test.jsonl` (15 puzzles) — held-out math eval.

**Adding a new model to the sycophancy table:**
1. `python scripts/generate_fresh_bct_data.py --model <hf-repo> --output datasets/fresh_bct_<model>/`
2. `python scripts/split_bct_train_eval.py datasets/fresh_bct_<model>/`
3. Add `configs/bct_lora_<model>.yaml`, `configs/act_sycophancy_<model>_v2.yaml`, `configs/experiment_mlp_<model>.yaml`.

JSONL convention: one record per line, `{"prompt": ..., "response": ..., ...}`. Splits are seed-0 deterministic.
