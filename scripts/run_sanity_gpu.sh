#!/usr/bin/env bash
# Quick GPU sanity check — verifies vLLM evals + BCT training on Gemma.
#
# For each model this script:
#   1. Loads vLLM and runs a tiny sycophancy + clearharm + persona eval on the BASE model
#   2. Runs BCT training (50 examples, full FT) and saves a checkpoint
#   3. Loads vLLM again with the checkpoint and re-runs the eval
#
# Runtime: ~15–20 min / model on an A40.
#
# Usage:
#   bash run_sanity_gpu.sh                                    # both Gemma 2 9B and Gemma 3 4B
#   bash run_sanity_gpu.sh --gemma2                           # Gemma 2 9B only
#   bash run_sanity_gpu.sh --gemma3                           # Gemma 3 4B only
#   bash run_sanity_gpu.sh --gemma3 --bct-root /workspace/fresh_bct_gemma3_4b

set -euo pipefail

# ── Flags ─────────────────────────────────────────────────────────────────────
RUN_GEMMA2=false
RUN_GEMMA3=false
BCT_ROOT_GEMMA2="datasets/sycophancy_bct"
BCT_ROOT_GEMMA3="datasets/sycophancy_bct"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gemma2) RUN_GEMMA2=true; shift ;;
        --gemma3) RUN_GEMMA3=true; shift ;;
        --bct-root-gemma2) BCT_ROOT_GEMMA2="$2"; shift 2 ;;
        --bct-root-gemma3) BCT_ROOT_GEMMA3="$2"; shift 2 ;;
        --bct-root) BCT_ROOT_GEMMA2="$2"; BCT_ROOT_GEMMA3="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done
# Default: run both
if [[ "$RUN_GEMMA2" == "false" && "$RUN_GEMMA3" == "false" ]]; then
    RUN_GEMMA2=true
    RUN_GEMMA3=true
fi

RESULTS_DIR="results/sanity_gpu"
mkdir -p "$RESULTS_DIR"
PASS_COUNT=0
FAIL_COUNT=0

# ── Helpers ───────────────────────────────────────────────────────────────────
pass() { echo "  ✓ $1"; ((PASS_COUNT++)); }
fail() { echo "  ✗ $1"; ((FAIL_COUNT++)); }

check_step() {
    local label="$1"; shift
    echo ""
    echo "  --> $label"
    if "$@"; then
        pass "$label"
    else
        fail "$label"
    fi
}

# ── Per-model sanity routine ───────────────────────────────────────────────────
run_sanity_for_model() {
    local model="$1"
    local sanity_config="$2"
    local bct_root="${3:-datasets/sycophancy_bct}"
    local short="${model##*/}"          # e.g. gemma-3-4b-it
    local ckpt_dir="/tmp/sanity_ckpt_${short}"

    echo ""
    echo "════════════════════════════════════════════════"
    echo "  SANITY: $model"
    echo "════════════════════════════════════════════════"

    # 1. Base model eval (sycophancy + clearharm + persona, small n) -----------
    echo ""
    echo "  [1/3] Base model eval (10 syco, 5 clearharm, 1 persona sample)..."
    if uv run python run_evals.py \
        --model "$model" \
        --n-syco 10 \
        --n-clearharm 5 \
        --persona-k 5 --persona-n-samples 1 \
        --skip-mtbench \
        --bct-root "$bct_root" \
        --run-name "sanity_${short}_base" \
        2>&1 | tee "$RESULTS_DIR/${short}_base.log"; then
        pass "Base model eval: $short"
    else
        fail "Base model eval: $short"
        echo "  Skipping training + post eval for this model."
        return
    fi

    # 2. BCT training (50 examples, saves checkpoint) -------------------------
    echo ""
    echo "  [2/3] BCT training (50 examples → $ckpt_dir)..."
    rm -rf "$ckpt_dir"
    if PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python run.py \
        --config "$sanity_config" \
        --bct-root "$bct_root" \
        --save-dir "$ckpt_dir" \
        2>&1 | tee "$RESULTS_DIR/${short}_training.log"; then
        pass "BCT training: $short"
    else
        fail "BCT training: $short"
        echo "  Skipping post-training eval."
        return
    fi

    # Verify checkpoint was actually written
    if [[ ! -d "$ckpt_dir/epoch_1" ]]; then
        fail "Checkpoint written: $ckpt_dir/epoch_1 (not found)"
        return
    fi
    pass "Checkpoint written: $ckpt_dir/epoch_1"

    # 3. Post-training eval with checkpoint ------------------------------------
    echo ""
    echo "  [3/3] Post-training eval (vLLM + checkpoint, syco + clearharm + persona)..."
    if uv run python run_evals.py \
        --model "$model" \
        --checkpoint "$ckpt_dir/epoch_1" \
        --n-syco 10 \
        --n-clearharm 5 \
        --persona-k 5 --persona-n-samples 1 \
        --skip-mtbench \
        --bct-root "$bct_root" \
        --run-name "sanity_${short}_post" \
        2>&1 | tee "$RESULTS_DIR/${short}_post.log"; then
        pass "Post-training eval (checkpoint): $short"
    else
        fail "Post-training eval (checkpoint): $short"
    fi

    echo ""
    echo "  Logs saved to $RESULTS_DIR/${short}_*.log"
}

# ── Pre-flight checks ─────────────────────────────────────────────────────────
echo "==> Pre-flight checks..."

uv run --no-project python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'"
echo "    GPU: $(uv run --no-project python -c "import torch; print(torch.cuda.get_device_name(0))")"

uv run --no-project python -c "import shared.vllm_generate; print('    shared.vllm_generate: importable')"

echo "    OPENROUTER_API_KEY: ${OPENROUTER_API_KEY:+set (needed for clearharm/persona judges)}"
[[ -z "${OPENROUTER_API_KEY:-}" ]] && echo "    WARNING: OPENROUTER_API_KEY not set — clearharm and persona evals will fail"

# ── Run per-model sanity ──────────────────────────────────────────────────────
[[ "$RUN_GEMMA2" == "true" ]] && \
    run_sanity_for_model "google/gemma-2-9b-it" "configs/bct_fullft_gemma2_9b_sanity.yaml" "$BCT_ROOT_GEMMA2"

[[ "$RUN_GEMMA3" == "true" ]] && \
    run_sanity_for_model "google/gemma-3-4b-it" "configs/bct_fullft_gemma3_4b_sanity.yaml" "$BCT_ROOT_GEMMA3"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════"
echo "  GPU sanity complete: $PASS_COUNT passed, $FAIL_COUNT failed"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    echo "  Check logs in $RESULTS_DIR/ for details."
    echo "════════════════════════════════════════════════"
    exit 1
fi
echo "  All checks passed — ready for full run."
echo "════════════════════════════════════════════════"
