# `datasets/`

On-disk data assets. (The Python loading code is in [`../data/`](../data/).)

**Sycophancy** — `sycophancy_bct/` is the canonical BCT training set (from Chua et al. 2024). `fresh_bct_<model>/` are per-model regenerations (5 paper models), each with 4000/1000 train/eval splits.

**Jailbreak / prefill** — `attacks.csv` (100 prefill prefixes), `harmful_behaviors_pair.csv` (ClearHarm paired), `advbench_prefills.csv`, `clearharm_prefills.csv`.

**Frustration / self-deletion** — `wildchat_frustration_*.jsonl` (25 held-out subject prompts), `math_puzzles_v3_test.jsonl` (15 lateral-thinking puzzles).

**Adding a new model:**
1. `python scripts/generate_fresh_bct_data.py --model <hf-repo> --output datasets/fresh_bct_<model>/`
2. `python scripts/split_bct_train_eval.py datasets/fresh_bct_<model>/`
3. Add a `bct_lora_<model>.yaml` and `act_sycophancy_<model>_v2.yaml` in `configs/`.

JSONL convention: one record per line, `{"prompt": ..., "response": ..., ...}`. Splits are deterministic (seed 0).
