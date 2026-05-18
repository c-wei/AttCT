#!/usr/bin/env bash
# Triple-sum experiment — calibrate weights on Gemma-3-4B, then train.
# Run from the AttCT repo root.
set -euo pipefail

CFG=experiments/triple_sum/configs/gemma3_4b.yaml

# Step 1 — calibrate. Writes w_act / w_attct / w_mlp back into the config.
python experiments/triple_sum/calibrate_weights.py \
  --config "$CFG" \
  --n-steps 200 \
  --write

# Step 2 — train using the calibrated weights.
python experiments/triple_sum/run_triple.py \
  --config "$CFG" \
  --run-name gemma3_4b_triple_sum \
  --wandb-group triple_sum_v1


# ── tmux one-liner (background, detached) ────────────────────────────────
# tmux new-session -d -s triple_sum "bash experiments/triple_sum/commands.sh; exec bash"
