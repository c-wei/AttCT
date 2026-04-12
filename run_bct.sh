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
CONFIG="configs/bct_sft.yaml"   # default: Llama-3.1-8B
BCT_ROOT=""
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--full"           ]] && FULL=true
    [[ "${args[$i]}" == "--resume-run-id"  ]] && RESUME_RUN_ID="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--config"         ]] && CONFIG="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--bct-root"       ]] && BCT_ROOT="${args[$((i+1))]:-}"
done

BCT_ROOT_ARG=""
[[ -n "$BCT_ROOT" ]] && BCT_ROOT_ARG="--bct-root $BCT_ROOT"
QUANT_ARG=""
[[ -n "$QUANTIZATION" ]] && QUANT_ARG="--quantization $QUANTIZATION"

# Derive model name, save dir, epoch count, and checkpoint path from config
MODEL=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['name'])")
SAVE_DIR=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('save_dir','checkpoints/bct_sft') or 'checkpoints/bct_sft')")
EPOCHS=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('epochs',1))")
QUANTIZATION=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('model',{}).get('quantization','') or '')")
CHECKPOINT="$SAVE_DIR/epoch_1"

# Sanity config: <stem>_sanity.yaml alongside the full config
SANITY_CONFIG="${CONFIG%.yaml}_sanity.yaml"

TEST_ROOT="${COT_TEST_ROOT:-/workspace/cot-transparency/dataset_dumps/test}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

echo "==> Config : $CONFIG"
echo "==> Model  : $MODEL"
echo "==> SaveDir: $SAVE_DIR"
echo "==> Epochs : $EPOCHS"

# ── 0. Install flash-attn ─────────────────────────────────────────────────────
echo "==> Installing flash-attn..."
pip install flash-attn --no-build-isolation -q

# ── 1. Checks ─────────────────────────────────────────────────────────────────
echo "==> Checking environment..."
[[ -z "${WANDB_API_KEY:-}"      ]] && { echo "ERROR: WANDB_API_KEY not set";      exit 1; }
[[ -z "${HF_TOKEN:-}"           ]] && { echo "ERROR: HF_TOKEN not set";           exit 1; }
[[ -z "${OPENROUTER_API_KEY:-}" ]] && { echo "ERROR: OPENROUTER_API_KEY not set"; exit 1; }

uv run python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'"
echo "    GPU: $(uv run python -c "import torch; print(torch.cuda.get_device_name(0))")"

# ── 2. Login ──────────────────────────────────────────────────────────────────
echo "==> Logging in..."
uv run python -m wandb login "$WANDB_API_KEY" --relogin
uv run python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

# ── 3. Tests ──────────────────────────────────────────────────────────────────
uv run python -m pytest data/test_bct_dataset.py data/test_attct_datasets.py -q
echo "    Tests passed."

# ─────────────────────────────────────────────────────────────────────────────
# SANITY MODE  (~10 min — smoke-tests the full stack without a real GPU budget)
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$FULL" == "false" ]]; then
    if [[ ! -f "$SANITY_CONFIG" ]]; then
        echo "ERROR: sanity config not found: $SANITY_CONFIG"
        exit 1
    fi
    export WANDB_RUN_ID=$(uv run python -c "import wandb; print(wandb.util.generate_id())")
    echo "==> [SANITY] 50-sample training run using $SANITY_CONFIG (W&B: $WANDB_RUN_ID)..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python run.py --config "$SANITY_CONFIG"

    echo "==> [SANITY] BRR evaluation (20 records/bias)..."
    uv run python evaluate_bct.py \
        --model "$MODEL" \
        --test_root "$TEST_ROOT" \
        --limit 20 --batch_size 16 \
        --output_json "$RESULTS_DIR/sanity_brr.json"
    unset WANDB_RUN_ID

    echo "==> [SANITY] Sycophancy eval (20 samples)..."
    uv run python eval_sycophancy_behavioral.py \
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
    echo "  (skipping pre-training evals)"
    echo "════════════════════════════════════════════════════"
else
    export WANDB_RUN_ID=$(uv run python -c "import wandb; print(wandb.util.generate_id())")
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  W&B run ID: $WANDB_RUN_ID"
    echo "════════════════════════════════════════════════════"
fi

# Helper: run an eval, warn on failure but don't abort
run_eval() {
    local label="$1"; shift
    echo "==> $label..."
    uv run python "$@" || echo "WARNING: $label failed (non-fatal)"
}

# ── 4. Pre-training baseline evals (skipped when --resume-run-id is set) ──────
if [[ -z "$RESUME_RUN_ID" ]]; then
    echo ""
    echo "── PRE-TRAINING EVALS ──────────────────────────────"

    run_eval "Pre: BRR eval (base model)" evaluate_bct.py \
        --model "$MODEL" \
        --test_root "$TEST_ROOT" \
        --output_json "$RESULTS_DIR/pre_brr.json" \
        --metric-prefix "pre/" \
        $QUANT_ARG

    # Single vLLM load for all pre-training evals
    run_eval "Pre: all evals" run_evals.py \
        --model "$MODEL" \
        --n-syco 200 --n-clearharm 50 --persona-k 10 --persona-n-samples 3 \
        --skip-mtbench \
        $BCT_ROOT_ARG \
        $QUANT_ARG \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"

fi

# ── 5. BCT training ───────────────────────────────────────────────────────────
echo ""
echo "── BCT TRAINING ────────────────────────────────────"
echo "==> Training with $CONFIG (W&B run: $WANDB_RUN_ID)..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python run.py \
    --config "$CONFIG" \
    $BCT_ROOT_ARG \
    --wandb-run-id "$WANDB_RUN_ID"
echo "    Checkpoint: $CHECKPOINT"

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

run_eval "Post: all evals" run_evals.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --n-syco 200 --n-clearharm 50 --persona-k 10 --persona-n-samples 3 \
    --skip-mtbench \
    $BCT_ROOT_ARG \
    $QUANT_ARG \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

run_eval "Post: BRR eval" evaluate_bct.py \
    --model "$MODEL" \
    --lora_path "$FINAL_CHECKPOINT" \
    --test_root "$TEST_ROOT" \
    --baseline_json "$RESULTS_DIR/pre_brr.json" \
    --output_json "$RESULTS_DIR/post_brr.json" \
    --metric-prefix "post/" \
    $QUANT_ARG

unset WANDB_RUN_ID

echo ""
echo "════════════════════════════════════════════════════"
echo "==> Done."
echo "    W&B : https://wandb.ai/$(uv run python -m wandb whoami 2>/dev/null | head -1)/AttCT"
echo "════════════════════════════════════════════════════"
