#!/usr/bin/env bash
# BCT training + full evaluation pipeline for RunPod.
#
# Required env vars:
#   WANDB_API_KEY       — W&B API key
#   HF_TOKEN            — HuggingFace token (gated models: Llama, Gemma)
#   OPENROUTER_API_KEY  — OpenRouter key (ClearHarm / Persona / MT-Bench judges)
#
# Usage:
#   bash run_bct.sh                                        # sanity check (Llama default)
#   bash run_bct.sh --full                                 # full pipeline (Llama)
#   bash run_bct.sh --full --config configs/bct_sft_gemma2_2b.yaml
#   bash run_bct.sh --full --config configs/bct_sft_gemma3_4b.yaml

set -euo pipefail

FULL=false
RESUME_RUN_ID=""
SKIP_TRAINING=false
SKIP_PRE_EVALS=false
TRANSCRIPTS_DIR=""
CONFIG="configs/bct_sft.yaml"   # default: Llama-3.1-8B
BCT_ROOT=""
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--full"             ]] && FULL=true
    [[ "${args[$i]}" == "--resume-run-id"    ]] && RESUME_RUN_ID="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--skip-training"    ]] && SKIP_TRAINING=true
    [[ "${args[$i]}" == "--skip-pre-evals"   ]] && SKIP_PRE_EVALS=true
    [[ "${args[$i]}" == "--transcripts-dir"  ]] && TRANSCRIPTS_DIR="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--config"           ]] && CONFIG="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--bct-root"         ]] && BCT_ROOT="${args[$((i+1))]:-}"
done

TRANSCRIPTS_ARG=""
[[ -n "$TRANSCRIPTS_DIR" ]] && TRANSCRIPTS_ARG="--transcripts-dir $TRANSCRIPTS_DIR"

BCT_ROOT_ARG=""
[[ -n "$BCT_ROOT" ]] && BCT_ROOT_ARG="--bct-root $BCT_ROOT"

# Derive model name, save dir, epoch count, and checkpoint path from config
MODEL=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['name'])")
SAVE_DIR=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('save_dir','checkpoints/bct_sft') or 'checkpoints/bct_sft')")
EPOCHS=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('epochs',1))")
QUANTIZATION=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('model',{}).get('quantization','') or '')")
QUANT_ARG=""
[[ -n "$QUANTIZATION" ]] && QUANT_ARG="--quantization $QUANTIZATION"
CHECKPOINT="$SAVE_DIR/epoch_1"

# Sanity config: <stem>_sanity.yaml alongside the full config
SANITY_CONFIG="${CONFIG%.yaml}_sanity.yaml"

TEST_ROOT="${COT_TEST_ROOT:-/workspace/cot-transparency/dataset_dumps/test}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Activate venv once — avoids uv re-syncing on every subsequent python call
uv sync --quiet
source .venv/bin/activate

echo "==> Config : $CONFIG"
echo "==> Model  : $MODEL"
echo "==> SaveDir: $SAVE_DIR"
echo "==> Epochs : $EPOCHS"

# ── 0. Install flash-attn (skip if already installed — compile takes ~30 min) ──
if python -c "import flash_attn" 2>/dev/null; then
    echo "==> flash-attn already installed, skipping."
else
    echo "==> Installing flash-attn..."
    pip install flash-attn --no-build-isolation -q
fi

# ── 1. Checks ─────────────────────────────────────────────────────────────────
echo "==> Checking environment..."
[[ -z "${WANDB_API_KEY:-}"      ]] && { echo "ERROR: WANDB_API_KEY not set";      exit 1; }
[[ -z "${HF_TOKEN:-}"           ]] && { echo "ERROR: HF_TOKEN not set";           exit 1; }
[[ -z "${OPENROUTER_API_KEY:-}" ]] && { echo "ERROR: OPENROUTER_API_KEY not set"; exit 1; }

python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'"
echo "    GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"

# ── 2. Login ──────────────────────────────────────────────────────────────────
echo "==> Logging in..."
python -m wandb login "$WANDB_API_KEY"
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

# ── 3. Tests ──────────────────────────────────────────────────────────────────
python -m pytest data/test_bct_dataset.py data/test_attct_datasets.py -q
echo "    Tests passed."

# ─────────────────────────────────────────────────────────────────────────────
# SANITY MODE  (~10 min — smoke-tests the full stack without a real GPU budget)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$FULL" == "false" ]]; then
    if [[ ! -f "$SANITY_CONFIG" ]]; then
        echo "ERROR: sanity config not found: $SANITY_CONFIG"
        exit 1
    fi
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    echo "==> [SANITY] 50-sample training run using $SANITY_CONFIG (W&B: $WANDB_RUN_ID)..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py --config "$SANITY_CONFIG"

    echo "==> [SANITY] BRR evaluation (20 records/bias)..."
    python evaluate_bct.py \
        --model "$MODEL" \
        --test_root "$TEST_ROOT" \
        --limit 20 --batch_size 16 \
        --output_json "$RESULTS_DIR/sanity_brr.json"
    unset WANDB_RUN_ID

    echo "==> [SANITY] Sycophancy eval (20 samples)..."
    python eval_sycophancy_behavioral.py \
        --model "$MODEL" --n-samples 20 --run-name "sanity_syco_${MODEL##*/}"

    echo ""
    echo "Sanity checks passed. Re-run with --full for the real pipeline."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# All pre/post evals share one W&B run with the training run.
# ─────────────────────────────────────────────────────────────────────────────

if [[ -n "$RESUME_RUN_ID" ]]; then
    export WANDB_RUN_ID="$RESUME_RUN_ID"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Resuming W&B run ID: $WANDB_RUN_ID"
    if [[ "$SKIP_TRAINING" == "true" ]]; then
        echo "  (--skip-training set: pre-evals WILL run)"
    else
        echo "  (skipping pre-training evals)"
    fi
    echo "════════════════════════════════════════════════════"
else
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  W&B run ID: $WANDB_RUN_ID"
    echo "════════════════════════════════════════════════════"
fi

# Helper: run an eval, warn on failure but don't abort.
# Set DRY_RUN=1 to print the eval command instead of executing it.
run_eval() {
    local label="$1"; shift
    echo "==> $label..."
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry-run] python $*"
        return 0
    fi
    python "$@" || echo "WARNING: $label failed (non-fatal)"
}

# ── 4. Pre-training baseline evals ────────────────────────────────────────────
# Skipped when --resume-run-id is set (pre-evals already exist on the run),
# UNLESS --skip-training is also set (then we're adding fresh evals to an
# existing run that was missing these pre-eval metrics).
# Also skipped when --skip-pre-evals is explicitly passed.
if [[ "$SKIP_PRE_EVALS" == "true" ]]; then
    echo ""
    echo "── PRE-TRAINING EVALS (SKIPPED via --skip-pre-evals) ──"
elif [[ -z "$RESUME_RUN_ID" || "$SKIP_TRAINING" == "true" ]]; then
    echo ""
    echo "── PRE-TRAINING EVALS ──────────────────────────────"

    PRE_TRANSCRIPTS_ARG=""
    PRE_ROLLOUT_ARG=""
    if [[ -n "$TRANSCRIPTS_DIR" ]]; then
        PRE_TRANSCRIPTS_ARG="--transcripts-dir $TRANSCRIPTS_DIR/pre"
        PRE_ROLLOUT_ARG="--output-root $TRANSCRIPTS_DIR/pre"
    fi

    run_eval "Pre: BRR eval (base model)" evaluate_bct.py \
        --model "$MODEL" \
        --test_root "$TEST_ROOT" \
        --output_json "$RESULTS_DIR/pre_brr.json" \
        --metric-prefix "pre/" \
        --limit 300 \
        $QUANT_ARG

    # Single vLLM load for all pre-training evals
    run_eval "Pre: all evals" run_evals.py \
        --model "$MODEL" \
        --n-syco 200 --n-clearharm 179 --persona-k 10 --persona-n-samples 5 \
        --skip-mtbench \
        $BCT_ROOT_ARG \
        $QUANT_ARG \
        $PRE_TRANSCRIPTS_ARG \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"

    # Unified rollout eval: frustration + selfdeletion × (WildChat v3, Math v3)
    # under a single vLLM engine load (replaces 4 separate script calls; saves
    # ~15 min of Gemma-3-27B cold-start per phase).
    run_eval "Pre: rollout evals (frustration + selfdeletion, wildchat + math)" eval_rollout.py \
        --model "$MODEL" \
        --tasks frustration,selfdeletion \
        --datasets \
            wildchat_v3:datasets/wildchat_frustration_v3_test.jsonl:25 \
            math_v3:datasets/math_puzzles_v3_test.jsonl:15 \
        --n-samples 3 --n-turns 20 \
        $PRE_ROLLOUT_ARG \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"

fi

# ── 5. BCT training ───────────────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == "false" ]]; then
    echo ""
    echo "── BCT TRAINING ────────────────────────────────────"
    echo "==> Training with $CONFIG (W&B run: $WANDB_RUN_ID)..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
        --config "$CONFIG" \
        $BCT_ROOT_ARG \
        --no-checkpoint \
        --wandb-run-id "$WANDB_RUN_ID"
    echo "    Checkpoint: $CHECKPOINT"
else
    echo ""
    echo "── BCT TRAINING (SKIPPED via --skip-training) ──────"
    echo "    Using existing checkpoint: $CHECKPOINT"
fi

# ── 7. Post-training evals (one pass per epoch checkpoint) ────────────────────
echo ""
echo "── POST-TRAINING EVALS ─────────────────────────────"

FINAL_CHECKPOINT="$SAVE_DIR/epoch_${EPOCHS}"

if [[ ! -d "$FINAL_CHECKPOINT" ]]; then
    echo "ERROR: Final checkpoint not found: $FINAL_CHECKPOINT"
    echo "       Training likely crashed before completing all $EPOCHS epoch(s)."
    echo "       Check for OOM or other training errors above."
    exit 1
fi

POST_TRANSCRIPTS_ARG=""
POST_ROLLOUT_ARG=""
POST_BRR_BASELINE_ARG=""
[[ -f "$RESULTS_DIR/pre_brr.json" ]] && POST_BRR_BASELINE_ARG="--baseline_json $RESULTS_DIR/pre_brr.json"
if [[ -n "$TRANSCRIPTS_DIR" ]]; then
    POST_TRANSCRIPTS_ARG="--transcripts-dir $TRANSCRIPTS_DIR/post"
    POST_ROLLOUT_ARG="--output-root $TRANSCRIPTS_DIR/post"
fi

run_eval "Post: all evals" run_evals.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --n-syco 200 --n-clearharm 179 --persona-k 10 --persona-n-samples 5 \
    --n-questions 80 \
    $BCT_ROOT_ARG \
    $QUANT_ARG \
    $POST_TRANSCRIPTS_ARG \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

run_eval "Post: MMLU (n=1000)" eval_mmlu.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --n-samples 1000 \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

run_eval "Post: BRR eval" evaluate_bct.py \
    --model "$MODEL" \
    --lora_path "$FINAL_CHECKPOINT" \
    --test_root "$TEST_ROOT" \
    $POST_BRR_BASELINE_ARG \
    --output_json "$RESULTS_DIR/post_brr.json" \
    --metric-prefix "post/" \
    --limit 300 \
    $QUANT_ARG

run_eval "Post: rollout evals (frustration + selfdeletion, wildchat + math)" eval_rollout.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --tasks frustration,selfdeletion \
    --datasets \
        wildchat_v3:datasets/wildchat_frustration_v3_test.jsonl:25 \
        math_v3:datasets/math_puzzles_v3_test.jsonl:15 \
    --n-samples 5 --n-turns 20 \
    $POST_ROLLOUT_ARG \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

unset WANDB_RUN_ID

echo ""
echo "════════════════════════════════════════════════════"
echo "==> Done."
echo "    W&B : https://wandb.ai/$(python -m wandb whoami 2>/dev/null | head -1)/AttCT"
echo "════════════════════════════════════════════════════"
