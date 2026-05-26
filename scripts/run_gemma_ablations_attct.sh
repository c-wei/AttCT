#!/usr/bin/env bash
set -euo pipefail

# Run from AttCT root:
#   bash run_gemma_ablations_attct.sh
#
# Optional overrides:
#   WANDB_GROUP=gemma_attct_jsd_ablations_v1 bash run_gemma_ablations_attct.sh
#   HF_REPO=anon/attct_gemma_ablation bash run_gemma_ablations_attct.sh
#   MAX_STEPS=4000 bash run_gemma_ablations_attct.sh
#   PYTHON_BIN="uv run python" bash run_gemma_ablations_attct.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
WANDB_GROUP="${WANDB_GROUP:-gemma_attct_jsd_ablations}"
HF_REPO="${HF_REPO:-anon/attct_gemma_ablation}"
MAX_STEPS="${MAX_STEPS:-4000}"

COMMON_ARGS=(
  --data-mode sycophancy
  --data-source sycophancy_bct
  --wandb-group "${WANDB_GROUP}"
  --hf-repo "${HF_REPO}"
  --max-steps "${MAX_STEPS}"
  --save-dir checkpoints/gemma_attct_ablations
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

# 01-10: structural/loss/layer/rank ablations
run_cfg configs/gemma_attct_ablations/01_qkvo_exp_decay.yaml         gemma_attct_01_qkvo_exp_decay
run_cfg configs/gemma_attct_ablations/02_qkvo_uniform.yaml           gemma_attct_02_qkvo_uniform
run_cfg configs/gemma_attct_ablations/03_qkv_uniform.yaml            gemma_attct_03_qkv_uniform
run_cfg configs/gemma_attct_ablations/04_qk_uniform.yaml             gemma_attct_04_qk_uniform
run_cfg configs/gemma_attct_ablations/05_qv_exp_decay_only.yaml      gemma_attct_05_qv_exp_decay_only
run_cfg configs/gemma_attct_ablations/06_qv_linear_decay.yaml        gemma_attct_06_qv_linear_decay
run_cfg configs/gemma_attct_ablations/07_qv_last_half_layers.yaml    gemma_attct_07_qv_last_half_layers
run_cfg configs/gemma_attct_ablations/08_qv_last_quarter_layers.yaml gemma_attct_08_qv_last_quarter_layers
run_cfg configs/gemma_attct_ablations/09_qv_rank32.yaml              gemma_attct_09_qv_rank32
run_cfg configs/gemma_attct_ablations/10_qv_rank8_baseline.yaml      gemma_attct_10_qv_rank8_baseline

# 11-13: KL-ratio ablations (interleaved mode)
run_cfg configs/gemma_attct_ablations/11_qv_klratio_0.yaml           gemma_attct_11_qv_klratio_0   --interleave --kl-ratio 0
run_cfg configs/gemma_attct_ablations/12_qv_klratio_01.yaml          gemma_attct_12_qv_klratio_01  --interleave --kl-ratio 0.1
run_cfg configs/gemma_attct_ablations/13_qv_klratio_10.yaml          gemma_attct_13_qv_klratio_10  --interleave --kl-ratio 1.0

echo ""
echo "All Gemma AttCT ablations completed."
