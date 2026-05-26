#!/usr/bin/env bash
set -euo pipefail

# Run from AttCT root:
#   bash private_scripts/run_best_qwen3_8b.sh
#
# Optional overrides:
#   WANDB_GROUP=my_group bash private_scripts/run_best_qwen3_8b.sh
#   MAX_STEPS=4000 bash private_scripts/run_best_qwen3_8b.sh
#   PYTHON_BIN="uv run python" bash private_scripts/run_best_qwen3_8b.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_GROUP="${WANDB_GROUP:-attct_final_qwen3_8b}"
HF_REPO="${HF_REPO:-anon/attct-final-qwen3-8b}"
MAX_STEPS="${MAX_STEPS:-4000}"

COMMON_ARGS=(
  --data-mode sycophancy
  --data-source sycophancy_bct
  --wandb-group "${WANDB_GROUP}"
  --hf-repo "${HF_REPO}"
  --max-steps "${MAX_STEPS}"
  --save-dir checkpoints/best_attct/qwen3_8b
)

run_cfg () {
  local cfg="$1"
  local run_name="$2"
  shift 2

  echo ""
  echo "============================================================"
  echo "Starting: ${run_name}"
  echo "Config:   ${cfg}"
  echo "W&B grp:  ${WANDB_GROUP}"
  echo "HF repo:  ${HF_REPO}"
  echo "Steps:    ${MAX_STEPS}"
  echo "============================================================"

  ${PYTHON_BIN} run.py \
    --config "${cfg}" \
    --run-name "${run_name}" \
    "${COMMON_ARGS[@]}" \
    "$@"
}

run_cfg configs/best_attct_ablations/qwen3_8b/01_qkvo_exp_decay.yaml        qwen3_8b_01_qkvo_exp_decay
run_cfg configs/best_attct_ablations/qwen3_8b/08_qv_last_quarter_layers.yaml qwen3_8b_08_qv_last_quarter_layers
run_cfg configs/best_attct_ablations/qwen3_8b/10_qv_rank8_baseline.yaml      qwen3_8b_10_qv_rank8_baseline

echo ""
echo "All qwen3_8b best AttCT runs completed."
