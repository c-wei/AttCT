# `private_scripts/`

Author-specific launchers for the paper's headline runs. Each invokes `run_act.sh` / `run_bct.sh` (or the jailbreak filter pipeline) with that model's best config and the W&B `RUN_GROUP` / `HF_REPO` conventions used in the paper.

- `run_best_<model>.sh` — five files, one per paper model (Gemma-3-4B, Gemma-3-27B, Llama-3.1-8B, Qwen3-4B, Qwen3-8B). Reproduces the within-threat headline row for that model.
- `run_jailbreak_gemma3_4b.sh` — paper's MLPCT-on-jailbreak run on Gemma-3-4B-IT via [`../experiments/jailbreak/run_jailbreak.sh`](../experiments/jailbreak/README.md).

Public reproductions: fork these and substitute your own `WANDB_GROUP` / `HF_REPO` env vars before running.
