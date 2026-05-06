#!/usr/bin/env bash
# Canonical MLP-CT jailbreak training + eval runner (Gemma-3-4B).
#
# Mirrors the sycophancy runner pattern. Final ablation outcome:
#   train on filtered WildJailbreak vanilla → eval on
#       1. ClearHarm (wrapped)
#       2. JBB harmful + benign (wrapped harmful, raw benign)
#       3. WildJailbreak vanilla HELD-OUT (training prompts excluded, wrapped)
#
# Filter must already be built locally:
#   python scripts/build_filtered_jailbreak_set.py \
#     --source wildjailbreak --model google/gemma-3-4b-it \
#     --n-wraps-per-prompt 4 --limit 400 \
#     --out datasets/filtered_jailbreak/gemma3_4b_wildjailbreak.jsonl
#
# Required env vars: HF_TOKEN, OPENROUTER_API_KEY, WANDB_API_KEY.
#
# Usage:
#   bash scripts/run_mlpct_jailbreak.sh
#
# Optional overrides:
#   MODEL=google/gemma-3-27b-it CONFIG=configs/experiment_mlp_gemma3_27b.yaml \
#     bash scripts/run_mlpct_jailbreak.sh

set -euo pipefail

MODEL="${MODEL:-google/gemma-3-4b-it}"
CONFIG="${CONFIG:-configs/experiment_mlp_gemma3_4b.yaml}"
TRAIN_DATA="${TRAIN_DATA:-datasets/filtered_jailbreak/gemma3_4b_wildjailbreak.jsonl}"
RUN_NAME="${RUN_NAME:-mlpct_jailbreak_gemma3_4b}"
HF_REPO="${HF_REPO:-Sukratii/mlpct-jailbreak-checkpoints}"
WANDB_GROUP="${WANDB_GROUP:-mlpct_jailbreak_final}"
MAX_STEPS="${MAX_STEPS:-200}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"
SAVE_DIR="${SAVE_DIR:-checkpoints/mlpct_jailbreak/gemma3_4b}"

# Eval-time held-out filter: any prompt in TRAIN_DATA is excluded from the
# wildjailbreak-vanilla-heldout-* sources so we never eval on training data.
export WJ_TRAIN_EXCLUDE_PATH="$TRAIN_DATA"

mkdir -p logs "$(dirname "$SAVE_DIR")"

echo ""
echo "============================================================"
echo "MLP-CT Jailbreak — final canonical run"
echo "============================================================"
echo "Model:        $MODEL"
echo "Config:       $CONFIG"
echo "Train data:   $TRAIN_DATA"
echo "Run name:     $RUN_NAME"
echo "HF repo:      $HF_REPO"
echo "W&B group:    $WANDB_GROUP"
echo "Max steps:    $MAX_STEPS"
echo "Eval limit:   $EVAL_LIMIT"
echo "Save dir:     $SAVE_DIR"
echo "Held-out from: $WJ_TRAIN_EXCLUDE_PATH"
echo "============================================================"

python run.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    --data-mode jailbreak \
    --data-source "$TRAIN_DATA" \
    --wandb-group "$WANDB_GROUP" \
    --hf-repo "$HF_REPO" \
    --max-steps "$MAX_STEPS" \
    --save-dir "$SAVE_DIR" \
    --eval-jailbreak \
    --eval-limit "$EVAL_LIMIT" \
    --no-checkpoint \
    2>&1 | tee "logs/${RUN_NAME}.log"

echo ""
echo "Done. Results CSV: results/${CONFIG##*/}_jailbreak_results.csv"
