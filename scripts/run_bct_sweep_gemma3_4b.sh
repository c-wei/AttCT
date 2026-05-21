#!/usr/bin/env bash
# LR sweep for Gemma 3 4B BCT — runs 4 configs sequentially.
#
# Configs:
#   1. Full FT  lr=1e-5  (paged_adamw_8bit)
#   2. Full FT  lr=5e-6  (paged_adamw_8bit)
#   3. LoRA     lr=1e-5
#   4. LoRA     lr=5e-6
#
# Usage:
#   bash run_bct_sweep_gemma3_4b.sh \
#       --bct-root /workspace/fresh_bct_gemma3_4b   # optional; uses config default if omitted
#
# Required env vars: WANDB_API_KEY, HF_TOKEN, OPENROUTER_API_KEY
# Optional: COT_TEST_ROOT

set -euo pipefail

BCT_ROOT=""
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--bct-root" ]] && BCT_ROOT="${args[$((i+1))]:-}"
done

BCT_ROOT_ARG=""
[[ -n "$BCT_ROOT" ]] && BCT_ROOT_ARG="--bct-root $BCT_ROOT"

CONFIGS=(
    "configs/bct_fullft_gemma3_4b_lr1e5.yaml"
    "configs/bct_fullft_gemma3_4b_lr5e6.yaml"
    "configs/bct_lora_gemma3_4b_lr1e5.yaml"
    "configs/bct_lora_gemma3_4b_lr5e6.yaml"
)

echo "════════════════════════════════════════════════════"
echo "  Gemma 3 4B BCT LR sweep — ${#CONFIGS[@]} configs"
echo "  BCT root: ${BCT_ROOT:-<from config>}"
echo "════════════════════════════════════════════════════"
echo ""

FAILED=()

for CONFIG in "${CONFIGS[@]}"; do
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Starting: $CONFIG"
    echo "════════════════════════════════════════════════════"
    echo ""

    if bash run_bct.sh --full --config "$CONFIG" $BCT_ROOT_ARG; then
        echo "==> DONE: $CONFIG"
    else
        echo "==> FAILED: $CONFIG (continuing sweep)"
        FAILED+=("$CONFIG")
    fi
done

echo ""
echo "════════════════════════════════════════════════════"
echo "  Sweep complete."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  FAILED configs:"
    for f in "${FAILED[@]}"; do echo "    - $f"; done
else
    echo "  All configs succeeded."
fi
echo "════════════════════════════════════════════════════"
