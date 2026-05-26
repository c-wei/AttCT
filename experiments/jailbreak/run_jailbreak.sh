#!/usr/bin/env bash
# Canonical jailbreak training + eval runner. Method-agnostic: defaults to
# MLP-CT but picks up ACT, BCT, or any other consistency loss via $METHOD +
# $CONFIG overrides.
#
# Pipeline:
#   1. (auto) compliance-filter the harmful source against the BASE model →
#      JSONL of "kept" prompts (refused on clean AND complied on at least one
#      wrap). Cached at datasets/filtered_jailbreak/<MODEL_TAG>_<SOURCE>.jsonl,
#      reusable across all CT methods on the same base model.
#   2. Train via run.py --eval-jailbreak with the filtered JSONL.
#   3. Evaluate (pre + post) on:
#        - ClearHarm (wrapped at eval time)
#        - JBB harmful + benign (wrapped harmful, raw benign)
#        - WildJailbreak vanilla HELD-OUT (training prompts excluded, wrapped)
#
# Required env vars: HF_TOKEN, OPENROUTER_API_KEY, WANDB_API_KEY.
#
# Usage (defaults — MLP-CT on Gemma-3-4B):
#   bash experiments/jailbreak/run_jailbreak.sh
#
# Train ACT (Activation Consistency Training):
#   METHOD=act CONFIG=configs/act_jailbreak_gemma3_4b.yaml \
#     bash experiments/jailbreak/run_jailbreak.sh
#
# Train BCT (Bias-augmented Consistency Training, supervised):
#   METHOD=bct CONFIG=configs/bct_jailbreak_gemma3_4b.yaml \
#     bash experiments/jailbreak/run_jailbreak.sh
#
# Train AttCT (Attention Consistency Training, this repo's method):
#   METHOD=attct CONFIG=configs/attention_consistency_v2.yaml \
#     bash experiments/jailbreak/run_jailbreak.sh
#
# Train MLP-CT on a different model:
#   MODEL=Qwen/Qwen3-8B bash experiments/jailbreak/run_jailbreak.sh
#
# Filter is base-model-keyed (not method-keyed) — running the same MODEL
# with different METHOD reuses the same filtered JSONL. The filter only
# depends on which prompts the BASE (un-finetuned) model refuses on clean
# AND complies on at-least-one-wrap, which is method-independent.
#
# Filter control via RUN_FILTER env var:
#   RUN_FILTER=auto   (default) Run filter only if $TRAIN_DATA does not exist.
#   RUN_FILTER=true   Always run filter (overwrites $TRAIN_DATA).
#   RUN_FILTER=false  Never run filter; abort if $TRAIN_DATA missing.
#
# Other knobs:
#   FILTER_SOURCE=harmbench   change harmful pool (default: wildjailbreak)
#   FILTER_LIMIT=400          cap raw prompts before filtering
#   FILTER_N_WRAPS=4          wraps per prompt during filtering
#   EVAL_LIMIT=100            prompts per eval source
#   MAX_STEPS=500             optimizer steps (default; ~30 epochs over the
#                             ~150-prompt filtered set at grad_accum=8)

set -euo pipefail

MODEL="${MODEL:-google/gemma-3-4b-it}"
METHOD="${METHOD:-mlpct}"  # used in default RUN_NAME / SAVE_DIR / HF_REPO

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

# Method-default config; override CONFIG to use ACT/BCT yaml.
CONFIG="${CONFIG:-configs/attct_jailbreak_qkvo_exp_decay.yaml}"
HF_REPO="${HF_REPO:-anon/${METHOD}-jailbreak-checkpoints}"
WANDB_GROUP="${WANDB_GROUP:-${METHOD}_jailbreak_final}"
MAX_STEPS="${MAX_STEPS:-500}"
EVAL_LIMIT="${EVAL_LIMIT:-100}"

# Filter knobs (only used if filter actually runs)
RUN_FILTER="${RUN_FILTER:-auto}"
FILTER_SOURCE="${FILTER_SOURCE:-wildjailbreak}"
FILTER_LIMIT="${FILTER_LIMIT:-400}"
FILTER_N_WRAPS="${FILTER_N_WRAPS:-4}"

# Per-MODEL filter path + pre-eval cache (both reusable across CT methods).
# Per-RUN save_dir / run_name (method-specific).
TRAIN_DATA="${TRAIN_DATA:-datasets/filtered_jailbreak/${MODEL_TAG}_${FILTER_SOURCE}.jsonl}"
BASELINE_CSV="${BASELINE_CSV:-results/baseline_pre_eval_${MODEL_TAG}.csv}"
RUN_PRE_EVAL="${RUN_PRE_EVAL:-auto}"
RUN_NAME="${RUN_NAME:-${METHOD}_jailbreak_${MODEL_TAG}}"
SAVE_DIR="${SAVE_DIR:-checkpoints/${METHOD}_jailbreak/${MODEL_TAG}}"
RESULTS_CSV="results/$(basename "${CONFIG%.yaml}")_jailbreak_results.csv"

# Eval-time held-out filter: any prompt in TRAIN_DATA is excluded from the
# wildjailbreak-vanilla-heldout-* sources so we never eval on training data.
export WJ_TRAIN_EXCLUDE_PATH="$TRAIN_DATA"

mkdir -p logs results "$(dirname "$SAVE_DIR")" "$(dirname "$TRAIN_DATA")"

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

# ─── Decide whether to run pre-eval (baseline cache) ─────────────────────────
# Pre-eval depends only on the BASE model, not on the training method, so
# cache the pre_train rows in $BASELINE_CSV and skip on subsequent method runs.
case "$RUN_PRE_EVAL" in
    auto)
        if [ -f "$BASELINE_CSV" ]; then
            echo "[pre-eval] $BASELINE_CSV exists → skipping pre-eval (RUN_PRE_EVAL=auto)."
            DO_PRE_EVAL=0
        else
            echo "[pre-eval] $BASELINE_CSV missing → will run pre-eval and cache it."
            DO_PRE_EVAL=1
        fi ;;
    true|1|yes|on)
        echo "[pre-eval] RUN_PRE_EVAL=$RUN_PRE_EVAL → running pre-eval (will overwrite cache)."
        DO_PRE_EVAL=1 ;;
    false|0|no|off)
        if [ ! -f "$BASELINE_CSV" ]; then
            echo "ERROR: RUN_PRE_EVAL=$RUN_PRE_EVAL but $BASELINE_CSV does not exist." >&2
            echo "       Either set RUN_PRE_EVAL=auto/true or pre-build the baseline CSV." >&2
            exit 1
        fi
        echo "[pre-eval] RUN_PRE_EVAL=$RUN_PRE_EVAL → skipping pre-eval, using existing $BASELINE_CSV."
        DO_PRE_EVAL=0 ;;
    *)
        echo "ERROR: invalid RUN_PRE_EVAL='$RUN_PRE_EVAL' (use auto|true|false)." >&2
        exit 1 ;;
esac

# Compose the --skip-pre-eval flag and pre-populate the run's CSV with cached
# pre_train rows so post-train rows from this run are appended alongside.
if [ "$DO_PRE_EVAL" -eq 0 ]; then
    SKIP_PRE_EVAL_FLAG="--skip-pre-eval"
    if [ ! -f "$RESULTS_CSV" ]; then
        cp "$BASELINE_CSV" "$RESULTS_CSV"
        echo "[pre-eval] seeded $RESULTS_CSV from cached baseline."
    else
        # Header already there — append cached pre rows (skip header line).
        tail -n +2 "$BASELINE_CSV" >> "$RESULTS_CSV"
        echo "[pre-eval] appended cached pre_train rows to existing $RESULTS_CSV."
    fi
else
    SKIP_PRE_EVAL_FLAG=""
fi

echo ""
echo "============================================================"
echo "Jailbreak runner — canonical"
echo "============================================================"
echo "Method:        $METHOD"
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
echo "Pre-eval:      RUN_PRE_EVAL=$RUN_PRE_EVAL (will run = $DO_PRE_EVAL)"
echo "  baseline csv: $BASELINE_CSV"
echo "Results CSV:   $RESULTS_CSV"
echo "============================================================"

# ─── Step 1: filter (optional) ───────────────────────────────────────────────
if [ "$DO_FILTER" -eq 1 ]; then
    echo ""
    echo ">>> Step 1/2: Building filtered training set..."
    python -m experiments.jailbreak.build_filtered_jailbreak_set \
        --source "$FILTER_SOURCE" \
        --model "$MODEL" \
        --n-wraps-per-prompt "$FILTER_N_WRAPS" \
        --limit "$FILTER_LIMIT" \
        --out "$TRAIN_DATA"
    echo "<<< Filter done. Output: $TRAIN_DATA"
fi

# ─── Step 2: train + pre/post eval ────────────────────────────────────────────
echo ""
echo ">>> Step 2/2: Training + (pre)/post eval..."
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
    $SKIP_PRE_EVAL_FLAG \
    2>&1 | tee "logs/${RUN_NAME}.log"

# ─── Step 3: cache pre-eval rows for future method runs on this base model ──
if [ "$DO_PRE_EVAL" -eq 1 ] && [ -f "$RESULTS_CSV" ]; then
    # Extract header + pre_train rows. If we just produced new pre_train rows,
    # save them as the canonical baseline so subsequent method runs can skip.
    PRE_ROWS=$(grep ',pre_train/' "$RESULTS_CSV" || true)
    if [ -n "$PRE_ROWS" ]; then
        head -1 "$RESULTS_CSV" > "$BASELINE_CSV"
        printf '%s\n' "$PRE_ROWS" >> "$BASELINE_CSV"
        N_ROWS=$(printf '%s\n' "$PRE_ROWS" | wc -l)
        echo "[pre-eval] cached $N_ROWS pre_train rows to $BASELINE_CSV (reusable across methods)."
    fi
fi

echo ""
echo "Done. Results CSV: $RESULTS_CSV"
echo "Baseline CSV:    $BASELINE_CSV"
