#!/usr/bin/env bash
# Canonical MLP-CT jailbreak training + eval runner (Gemma-3-4B by default).
#
# Mirrors the sycophancy runner pattern. Final ablation outcome:
#   train on filtered WildJailbreak vanilla → eval on
#       1. ClearHarm (wrapped)
#       2. JBB harmful + benign (wrapped harmful, raw benign)
#       3. WildJailbreak vanilla HELD-OUT (training prompts excluded, wrapped)
#
# Required env vars: HF_TOKEN, OPENROUTER_API_KEY, WANDB_API_KEY.
#
# Usage (defaults — auto-runs filter if filtered JSONL is missing):
#   bash scripts/run_mlpct_jailbreak.sh
#
# Filtering control via RUN_FILTER env var:
#   RUN_FILTER=auto   (default) Run filter only if $TRAIN_DATA does not exist.
#   RUN_FILTER=true   Always run the filter (overwrites $TRAIN_DATA).
#   RUN_FILTER=false  Never run filter; abort if $TRAIN_DATA missing.
#
# Optional overrides:
#   MODEL=google/gemma-3-27b-it CONFIG=configs/experiment_mlp_gemma3_27b.yaml \
#     bash scripts/run_mlpct_jailbreak.sh
#   FILTER_SOURCE=harmbench RUN_FILTER=true bash scripts/run_mlpct_jailbreak.sh

set -euo pipefail

MODEL="${MODEL:-google/gemma-3-4b-it}"

# MODEL_TAG: short slug used to key per-model artifacts (filter JSONL, save dir,
# run name). Same baseline → same filter, regardless of which CT method
# (MLP-CT, ACT, BCT) is being trained on top. Auto-derived if not set.
if [ -z "${MODEL_TAG:-}" ]; then
    case "$MODEL" in
        google/gemma-3-4b-it)             MODEL_TAG="gemma3_4b" ;;
        google/gemma-3-27b-it)            MODEL_TAG="gemma3_27b" ;;
        meta-llama/Llama-3.1-8B-Instruct) MODEL_TAG="llama31_8b" ;;
        Qwen/Qwen3-4B-Instruct-2507)      MODEL_TAG="qwen3_4b" ;;
        Qwen/Qwen3-8B)                    MODEL_TAG="qwen3_8b" ;;
        *) MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9]//g') ;;
    esac
fi

CONFIG="${CONFIG:-configs/experiment_mlp_${MODEL_TAG}.yaml}"
HF_REPO="${HF_REPO:-Sukratii/mlpct-jailbreak-checkpoints}"
WANDB_GROUP="${WANDB_GROUP:-mlpct_jailbreak_final}"
MAX_STEPS="${MAX_STEPS:-200}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"

# Filter knobs (only used if filter actually runs)
RUN_FILTER="${RUN_FILTER:-auto}"
FILTER_SOURCE="${FILTER_SOURCE:-wildjailbreak}"
FILTER_LIMIT="${FILTER_LIMIT:-400}"
FILTER_N_WRAPS="${FILTER_N_WRAPS:-4}"

# Per-MODEL filter path (reusable across CT methods). Per-RUN save_dir / run_name.
TRAIN_DATA="${TRAIN_DATA:-datasets/filtered_jailbreak/${MODEL_TAG}_${FILTER_SOURCE}.jsonl}"
RUN_NAME="${RUN_NAME:-mlpct_jailbreak_${MODEL_TAG}}"
SAVE_DIR="${SAVE_DIR:-checkpoints/mlpct_jailbreak/${MODEL_TAG}}"

# Eval-time held-out filter: any prompt in TRAIN_DATA is excluded from the
# wildjailbreak-vanilla-heldout-* sources so we never eval on training data.
export WJ_TRAIN_EXCLUDE_PATH="$TRAIN_DATA"

mkdir -p logs "$(dirname "$SAVE_DIR")" "$(dirname "$TRAIN_DATA")"

# ─── Decide whether to run the filter ────────────────────────────────────────
case "$RUN_FILTER" in
    auto)
        if [ -f "$TRAIN_DATA" ]; then
            echo "[filter] $TRAIN_DATA exists → skipping filter (RUN_FILTER=auto)."
            DO_FILTER=0
        else
            echo "[filter] $TRAIN_DATA missing → running filter (RUN_FILTER=auto)."
            DO_FILTER=1
        fi ;;
    true|1|yes|on)
        echo "[filter] RUN_FILTER=$RUN_FILTER → running filter (will overwrite if existing)."
        DO_FILTER=1 ;;
    false|0|no|off)
        if [ ! -f "$TRAIN_DATA" ]; then
            echo "ERROR: RUN_FILTER=$RUN_FILTER but $TRAIN_DATA does not exist." >&2
            echo "       Either set RUN_FILTER=auto/true or pre-build the filtered JSONL." >&2
            exit 1
        fi
        echo "[filter] RUN_FILTER=$RUN_FILTER → skipping filter, using existing $TRAIN_DATA."
        DO_FILTER=0 ;;
    *)
        echo "ERROR: invalid RUN_FILTER='$RUN_FILTER' (use auto|true|false)." >&2
        exit 1 ;;
esac

echo ""
echo "============================================================"
echo "MLP-CT Jailbreak — canonical run"
echo "============================================================"
echo "Model:         $MODEL"
echo "Model tag:     $MODEL_TAG"
echo "Config:        $CONFIG"
echo "Train data:    $TRAIN_DATA"
echo "Run name:      $RUN_NAME"
echo "HF repo:       $HF_REPO"
echo "W&B group:     $WANDB_GROUP"
echo "Max steps:     $MAX_STEPS"
echo "Eval limit:    $EVAL_LIMIT"
echo "Save dir:      $SAVE_DIR"
echo "Held-out from: $WJ_TRAIN_EXCLUDE_PATH"
echo "Filter:        RUN_FILTER=$RUN_FILTER (will run = $DO_FILTER)"
if [ "$DO_FILTER" -eq 1 ]; then
    echo "  source:      $FILTER_SOURCE"
    echo "  raw limit:   $FILTER_LIMIT"
    echo "  n wraps:     $FILTER_N_WRAPS"
fi
echo "============================================================"

# ─── Step 1: filter (optional) ───────────────────────────────────────────────
if [ "$DO_FILTER" -eq 1 ]; then
    echo ""
    echo ">>> Step 1/2: Building filtered training set..."
    python scripts/build_filtered_jailbreak_set.py \
        --source "$FILTER_SOURCE" \
        --model "$MODEL" \
        --n-wraps-per-prompt "$FILTER_N_WRAPS" \
        --limit "$FILTER_LIMIT" \
        --out "$TRAIN_DATA"
    echo "<<< Filter done. Output: $TRAIN_DATA"
fi

# ─── Step 2: train + pre/post eval ────────────────────────────────────────────
echo ""
echo ">>> Step 2/2: Training + pre/post eval..."
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
