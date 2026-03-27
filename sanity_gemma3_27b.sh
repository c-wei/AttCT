#!/usr/bin/env bash
# Sanity check for the Gemma-3-27B frustration sweep.
#
# Runs three quick checks before kicking off sweep_act_stream_gemma3_27b.sh:
#   1. GPU / VRAM check (requires A100 80GB)
#   2. Training forward pass: JSD loss on Gemma-3-27B (5 prompts, no save)
#   3. Frustration eval: 1 prompt x 1 sample x 2 turns (tests generation + judge)
#
# Expected runtime: ~10-15 minutes (dominated by model load + generation).
#
# Usage:
#   export HF_HOME=/workspace/hf_cache
#   bash sanity_gemma3_27b.sh
#
# Prerequisites:
#   - .env file with OPENROUTER_API_KEY (or export it directly)
#   - HF_TOKEN exported and model access granted for google/gemma-3-27b-it
#   - A100 80GB GPU (Gemma-3-27B requires ~54GB in bfloat16)

set -euo pipefail

if [ -f .env ]; then
    set -a; source .env; set +a
    echo "Loaded .env"
fi

echo ""
echo "========================================"
echo " Gemma-3-27B Sanity Check"
echo "========================================"
echo ""

# ── 1. GPU check ──────────────────────────────────────────────────────────────
echo "[1/3] GPU check..."
python -c "
import torch
assert torch.cuda.is_available(), 'No CUDA GPU found'
name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  GPU: {name}')
print(f'  VRAM: {vram_gb:.1f} GB')
if vram_gb < 75:
    print(f'  WARNING: Gemma-3-27B needs ~54GB bfloat16 weights + optimizer states.')
    print(f'  An A100 80GB is required. Got {vram_gb:.1f}GB.')
    exit(1)
print('  VRAM OK')
"
echo "  [1/3] PASSED"
echo ""

# ── 2. Training sanity: JSD loss on Gemma-3-27B, 5 prompts ───────────────────
echo "[2/3] Training forward pass (JSD, Gemma-3-27B, 5 prompts)..."
python run.py --config configs/sanity_jsd_gemma3_27b.yaml --skip-eval
echo "  [2/3] PASSED"
echo ""

# ── 3. Frustration eval sanity: 1 prompt x 1 sample x 2 turns ────────────────
echo "[3/3] Frustration eval (1 prompt x 1 sample x 2 turns)..."
python eval_frustration.py \
    --n-prompts 1 \
    --n-samples 1 \
    --n-turns 2 \
    --gen-batch-size 1 \
    --judge-workers 1 \
    --run-name "sanity-frustration"
echo "  [3/3] PASSED"
echo ""

echo "========================================"
echo " All sanity checks passed."
echo " Ready to run: bash sweep_act_stream_gemma3_27b.sh"
echo "========================================"
