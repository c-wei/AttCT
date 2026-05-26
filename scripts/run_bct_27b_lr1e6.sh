#!/usr/bin/env bash
# BCT training + post evals for Gemma 3 27B LoRA LR=1e-6.
#
# Pre-eval baseline already exists from the LR=5e-6 run — this script runs
# training + post evals only (~6h on A100 80GB).
#
# Usage:
#   bash run_bct_27b_lr1e6.sh
#
# Requires (either exported or present in /workspace/AttCT/.env):
#   WANDB_API_KEY, HF_TOKEN, OPENROUTER_API_KEY

set -euo pipefail

# ── Check env ─────────────────────────────────────────────────────────────────
[[ -z "${WANDB_API_KEY:-}"      ]] && { echo "ERROR: WANDB_API_KEY not set";      exit 1; }
[[ -z "${HF_TOKEN:-}"           ]] && { echo "ERROR: HF_TOKEN not set";           exit 1; }
[[ -z "${OPENROUTER_API_KEY:-}" ]] && { echo "ERROR: OPENROUTER_API_KEY not set"; exit 1; }

# ── Config ────────────────────────────────────────────────────────────────────
MODEL="google/gemma-3-27b-it"
CONFIG="configs/bct_lora_gemma3_27b_lr1e6.yaml"
CHECKPOINT="checkpoints/bct_lora_gemma3_27b_lr1e6/epoch_1"
TEST_ROOT="${COT_TEST_ROOT:-/workspace/cot-transparency/dataset_dumps/test}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# Pre-generate W&B run ID so training + all post evals share one run
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")

echo ""
echo "════════════════════════════════════════════════════"
echo "  Gemma 3 27B LoRA BCT — LR = 1e-6"
echo "  W&B run ID : $WANDB_RUN_ID"
echo "  Checkpoint : $CHECKPOINT"
echo "════════════════════════════════════════════════════"
echo ""

# ── Activate venv ─────────────────────────────────────────────────────────────
uv sync --quiet
source .venv/bin/activate

python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'"
echo "==> GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"

python -m wandb login "$WANDB_API_KEY"
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

# Helper: run an eval step, warn on failure but don't abort the whole pipeline
run_eval() {
    local label="$1"; shift
    echo ""
    echo "── $label ──────────────────────────────────────────"
    python "$@" || echo "WARNING: $label failed (non-fatal, continuing)"
}

# ── 1. Training ───────────────────────────────────────────────────────────────
echo ""
echo "── BCT TRAINING ─────────────────────────────────────"
echo "==> Training with $CONFIG..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
    --config "$CONFIG" \
    --no-checkpoint \
    --wandb-run-id "$WANDB_RUN_ID"

[[ -d "$CHECKPOINT" ]] || {
    echo "ERROR: checkpoint not found at $CHECKPOINT — training likely failed."
    exit 1
}
echo "==> Checkpoint ready: $CHECKPOINT"

# ── 2. Post: sycophancy + ClearHarm + persona + MT-Bench (one vLLM load) ─────
run_eval "Post: run_evals" run_evals.py \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --n-syco 200 --n-clearharm 50 \
    --persona-k 10 --persona-n-samples 3 \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

# ── 3. Post: BRR ─────────────────────────────────────────────────────────────
run_eval "Post: BRR eval" -m experiments.sycophancy.evaluate_bct \
    --model "$MODEL" \
    --lora_path "$CHECKPOINT" \
    --test_root "$TEST_ROOT" \
    --baseline_json "$RESULTS_DIR/pre_brr.json" \
    --output_json  "$RESULTS_DIR/post_brr_lr1e6.json" \
    --metric-prefix "post/" \
    --limit 300

# ── 4. Post: frustration (8-turn) ────────────────────────────────────────────
run_eval "Post: frustration (8-turn)" -m experiments.frustration.eval_frustration \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --prompts-file datasets/wildchat_frustration_prompts_v2.jsonl \
    --n-turns 8 --max-model-len 8192 \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

# ── 5. Post: self-deletion ────────────────────────────────────────────────────
run_eval "Post: selfdeletion" -m experiments.frustration.eval_selfdeletion \
    --model "$MODEL" \
    --checkpoint "$CHECKPOINT" \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

unset WANDB_RUN_ID

echo ""
echo "════════════════════════════════════════════════════"
echo "==> All done."
echo "    W&B: https://wandb.ai/<your-wandb-entity>/consistency-training-anon"
echo "════════════════════════════════════════════════════"
