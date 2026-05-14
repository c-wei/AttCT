# `private_scripts/`

Hand-tuned launchers for each paper-headline row. One per model, plus the jailbreak-specific filter-train-eval.

- `run_best_<model>.sh` — five files, one per paper model. Each invokes `run_act.sh` / `run_bct.sh` with the best config + hyperparameters for that row.
- `run_jailbreak_gemma3_4b.sh` — paper MLPCT-on-jailbreak run.

These contain W&B run-name and group conventions specific to the paper authors. Public reproductions should fork and substitute your own `WANDB_GROUP` / `HF_REPO`. Invoke from the repo root.
