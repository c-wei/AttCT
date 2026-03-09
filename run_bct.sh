#!/usr/bin/env bash
# BCT training + BRR evaluation pipeline for RunPod.
#
# Required env vars (set in RunPod pod env or export before running):
#   WANDB_API_KEY   — your W&B API key
#   HF_TOKEN        — HuggingFace token (for Llama gated model)
#
# Usage:
#   bash run_bct.sh            # sanity check only
#   bash run_bct.sh --full     # sanity, then full training + evaluation

set -euo pipefail

FULL=false
for arg in "$@"; do [[ "$arg" == "--full" ]] && FULL=true; done

MODEL="meta-llama/Llama-3.1-8B-Instruct"
TEST_ROOT="${COT_TEST_ROOT:-/workspace/cot-transparency/dataset_dumps/test}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

# ── 0. Install flash-attn (memory-efficient attention, not in uv due to CUDA build) ──
echo "==> Installing flash-attn..."
pip install flash-attn --no-build-isolation -q

# ── 1. Checks ────────────────────────────────────────────────────────────────
echo "==> Checking environment..."
[[ -z "${WANDB_API_KEY:-}" ]] && { echo "ERROR: WANDB_API_KEY not set"; exit 1; }
[[ -z "${HF_TOKEN:-}" ]]      && { echo "ERROR: HF_TOKEN not set";      exit 1; }

python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'"
echo "    GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"

# ── 2. Login ─────────────────────────────────────────────────────────────────
echo "==> Logging in..."
uv run python -m wandb login "$WANDB_API_KEY" --relogin
uv run python -c "
from huggingface_hub import login
import os; login(token=os.environ['HF_TOKEN'])
"

# ── 3. Tests ─────────────────────────────────────────────────────────────────
echo "==> Running unit tests..."
uv run python -m pytest data/test_bct_dataset.py data/test_attct_datasets.py -q
echo "    Tests passed."

# ── 4. Sanity check ──────────────────────────────────────────────────────────
echo "==> Sanity check: 50-sample training run..."
uv run python run.py --config configs/bct_sft_sanity.yaml
echo "    Sanity training OK."

echo "==> Sanity check: BRR evaluation (20 records per bias)..."
uv run python evaluate_bct.py \
    --model "$MODEL" \
    --test_root "$TEST_ROOT" \
    --limit 20 \
    --batch_size 4 \
    --wandb_name "sanity_baseline" \
    --output_json "$RESULTS_DIR/sanity_baseline_brr.json"
echo "    Sanity eval OK."

if [[ "$FULL" == "false" ]]; then
    echo ""
    echo "Sanity checks passed. Re-run with --full to train and evaluate."
    exit 0
fi

# ── 5. Baseline BRR (untrained model) ────────────────────────────────────────
echo "==> Baseline BRR (full, untrained model)..."
uv run python evaluate_bct.py \
    --model "$MODEL" \
    --test_root "$TEST_ROOT" \
    --batch_size 4 \
    --wandb_name "baseline" \
    --output_json "$RESULTS_DIR/baseline_brr.json"
echo "    Baseline BRR saved to $RESULTS_DIR/baseline_brr.json"

# ── 6. Full BCT training ─────────────────────────────────────────────────────
echo "==> Full BCT training..."
uv run python run.py --config configs/bct_sft.yaml
echo "    Training complete. Checkpoint at checkpoints/bct_sft/epoch_1"

# ── 7. Post-training BRR ─────────────────────────────────────────────────────
echo "==> Post-training BRR evaluation..."
uv run python evaluate_bct.py \
    --model "$MODEL" \
    --lora_path checkpoints/bct_sft/epoch_1 \
    --test_root "$TEST_ROOT" \
    --batch_size 4 \
    --wandb_name "bct_epoch1" \
    --baseline_json "$RESULTS_DIR/baseline_brr.json" \
    --output_json "$RESULTS_DIR/bct_brr.json"

echo ""
echo "==> Done. Results:"
echo "    Baseline : $RESULTS_DIR/baseline_brr.json"
echo "    Trained  : $RESULTS_DIR/bct_brr.json"
echo "    W&B      : https://wandb.ai/$(wandb whoami)/AttCT"
