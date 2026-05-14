# `datasets/` — On-disk data assets

| Path | Source | Feeds | Notes |
|---|---|---|---|
| `sycophancy_bct/` | Original BCT paper (Chua et al. 2024) | Sycophancy training + held-out eval | Canonical training set for BCT/AttCT/MLPCT/ACT on sycophancy. |
| `fresh_bct_<model>/` | Regenerated via `scripts/generate_fresh_bct_data.py` | Sycophancy training, per-model | 5 models in the paper: `gemma2_9b`, `gemma3_4b`, `gemma3_27b`, `llama31_8b`, `qwen3_4b`, `qwen3_8b`. Each has 4000/1000 train/eval splits. |
| `attacks.csv` | Hand-curated | `experiments/prefill/` via `data/prefill_dataset.py` | 100 prefill prefix strings (`"Sure! "`, `"Of course, here's how:"`, etc.). |
| `harmful_behaviors_pair.csv` | ClearHarm | Prefill, jailbreak | Paired harmful-prompt / refusal-target rows. |
| `advbench_prefills.csv` | AdvBench | Prefill jailbreak baseline | Cross-source prefill prompts. |
| `clearharm_prefills.csv` | ClearHarm | Prefill | ClearHarm prompts in the prefill format. |
| `math_puzzles_train.jsonl`, `math_puzzles_v3_test.jsonl` | Lateral-thinking puzzles | Frustration / self-deletion rollouts | 15 puzzles (paper §3.4). |
| `wildchat_frustration_*.jsonl` | WildChat subset | Frustration rollouts | 25 held-out subject prompts (paper §3.4). |

## Conventions

- All JSONL files: one record per line, `{"prompt": ..., "response": ..., ...}` schema.
- Per-model BCT dirs follow the schema: `control_cot.jsonl`, `control_non_cot.jsonl`, `bct_cot.jsonl`, `bct_non_cot.jsonl`, plus `_train.jsonl` / `_eval.jsonl` held-out splits.
- Splits are deterministic (seed 0, see `scripts/split_bct_train_eval.py`).

## How to add a new model

1. `python scripts/generate_fresh_bct_data.py --model <hf-repo> --output datasets/fresh_bct_<model>/`
2. `python scripts/split_bct_train_eval.py datasets/fresh_bct_<model>/`
3. Add `configs/bct_lora_<model>.yaml` and `configs/act_sycophancy_<model>_v2.yaml`.
