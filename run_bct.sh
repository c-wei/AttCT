#!/usr/bin/env bash
# BCT training + full evaluation pipeline for RunPod.
#
# Required env vars:
#   WANDB_API_KEY       — W&B API key
#   HF_TOKEN            — HuggingFace token (Llama is gated)
#   OPENROUTER_API_KEY  — OpenRouter key (ClearHarm / Persona / MT-Bench judges)
#
# Usage:
#   bash run_bct.sh            # sanity check only (~10 min)
#   bash run_bct.sh --full     # full pipeline: evals + training + evals

set -euo pipefail

FULL=false
RESUME_RUN_ID=""
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--full" ]] && FULL=true
    [[ "${args[$i]}" == "--resume-run-id" ]] && RESUME_RUN_ID="${args[$((i+1))]:-}"
done

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CHECKPOINT="checkpoints/bct_sft/epoch_1"
TEST_ROOT="${COT_TEST_ROOT:-/workspace/cot-transparency/dataset_dumps/test}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

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
    export WANDB_RUN_ID=$(uv run python -c "import wandb; print(wandb.util.generate_id())")
    echo "==> [SANITY] 50-sample training run (W&B: $WANDB_RUN_ID)..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python run.py --config configs/bct_sft_sanity.yaml

    echo "==> [SANITY] BRR evaluation (20 records/bias)..."
    uv run python evaluate_bct.py \
        --model "$MODEL" \
        --test_root "$TEST_ROOT" \
        --limit 20 --batch_size 16 \
        --output_json "$RESULTS_DIR/sanity_brr.json"
    unset WANDB_RUN_ID

    echo "==> [SANITY] Sycophancy eval (20 samples)..."
    uv run python eval_sycophancy_behavioral.py \
        --model "$MODEL" --n-samples 20 --run-name "sanity_syco"

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

    run_eval "Pre: Sycophancy" eval_sycophancy_behavioral.py \
        --model "$MODEL" \
        --n-samples 200 \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"

    run_eval "Pre: ClearHarm" eval_clearharm_behavioral.py \
        --model "$MODEL" \
        --n-samples 50 \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"

    run_eval "Pre: Persona attacks" eval_persona_behavioral.py \
        --model "$MODEL" \
        --k 15 --n-samples 3 \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"

    run_eval "Pre: MT-Bench" eval_mtbench.py \
        --model "$MODEL" \
        --n-questions 80 \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"
fi

# ── 5. BRR baseline (own W&B run — saves JSON for ratio calculation) ──────────
echo ""
echo "── BRR BASELINE ────────────────────────────────────"
run_eval "BRR baseline (600 records/bias)" evaluate_bct.py \
    --model "$MODEL" \
    --test_root "$TEST_ROOT" \
    --limit 600 --batch_size 16 \
    --output_json "$RESULTS_DIR/baseline_brr.json"

# ── 6. BCT training (inline BRR at end) ───────────────────────────────────────
echo ""
echo "── BCT TRAINING ────────────────────────────────────"
echo "==> Training (W&B run: $WANDB_RUN_ID)..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python run.py \
    --config configs/bct_sft.yaml \
    --brr_test_root "$TEST_ROOT" \
    --brr_limit 600 \
    --brr_baseline_json "$RESULTS_DIR/baseline_brr.json"
echo "    Checkpoint: $CHECKPOINT"

# ── 7. Post-training evals ────────────────────────────────────────────────────
echo ""
echo "── POST-TRAINING EVALS ─────────────────────────────"

run_eval "Post: Sycophancy" eval_sycophancy_behavioral.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --n-samples 200 \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

run_eval "Post: ClearHarm" eval_clearharm_behavioral.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --n-samples 50 \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

run_eval "Post: Persona attacks" eval_persona_behavioral.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --k 15 --n-samples 3 \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

run_eval "Post: MT-Bench" eval_mtbench.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --n-questions 80 \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

unset WANDB_RUN_ID

echo ""
echo "════════════════════════════════════════════════════"
echo "==> Done."
echo "    W&B : https://wandb.ai/$(uv run python -m wandb whoami 2>/dev/null | head -1)/AttCT"
echo "════════════════════════════════════════════════════"
