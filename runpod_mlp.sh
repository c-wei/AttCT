#!/usr/bin/env bash
# RunPod setup and training script for MLP Consistency Training.
# Run this after SSHing into a RunPod instance (A40 48GB or A100 80GB).
#
# STORAGE: Attach a network volume (>=30 GB) mounted at /workspace.
#
# Usage:
#   bash runpod_mlp.sh                          # sanity check only (tiny-gpt2)
#   bash runpod_mlp.sh --full                   # sanity + full training (LLaMA-8B)
#   bash runpod_mlp.sh --full --variant output  # full training, Variant B
#   bash runpod_mlp.sh --full --variant hidden  # full training, Variant A (default)
#
# Required environment variables:
#   HF_TOKEN   — HuggingFace token (needs Llama access for --full)
#
# Optional:
#   WANDB_KEY  — Weights & Biases API key (if unset, logging goes to console only)

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
FULL=false
VARIANT="hidden"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --full)      FULL=true; shift ;;
    --variant)   VARIANT="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "$VARIANT" != "hidden" && "$VARIANT" != "output" ]]; then
  echo "ERROR: --variant must be 'hidden' or 'output', got '$VARIANT'"
  exit 1
fi

REPO_URL="https://github.com/c-wei/AttCT.git"
BRANCH="sukratii-mlp"
WORKDIR="/workspace/AttCT-mlp"

# Map variant to configs
SANITY_CONFIG="configs/sanity_mlp_${VARIANT}.yaml"
FULL_CONFIG="configs/mlp_${VARIANT}_consistency.yaml"

# Redirect HuggingFace cache to network volume
export HF_HOME="/workspace/hf_cache"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# ── 1. Check required env vars ───────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Export it before running this script."
  exit 1
fi
if [[ -z "${WANDB_KEY:-}" ]]; then
  echo ">>> WANDB_KEY not set — W&B logging disabled (console only)."
  export WANDB_MODE="disabled"
fi

WORKSPACE_AVAIL_GB=$(df -BG /workspace | awk 'NR==2 {gsub("G",""); print $4}')
echo ">>> /workspace available: ${WORKSPACE_AVAIL_GB}GB"
echo ">>> Variant: ${VARIANT}"
echo ">>> Full training: ${FULL}"

# ── 2. Clone repo ────────────────────────────────────────────────────────────
if [[ ! -d "$WORKDIR" ]]; then
  echo ">>> Cloning repo (branch: $BRANCH)..."
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
else
  echo ">>> Repo already exists at $WORKDIR, pulling latest..."
  git -C "$WORKDIR" fetch origin
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" pull origin "$BRANCH"
fi

cd "$WORKDIR"

# ── 3. Install dependencies ──────────────────────────────────────────────────
echo ">>> Installing Python dependencies..."
pip install --quiet torch transformers peft datasets wandb tqdm pyyaml

# ── 4. Authenticate ──────────────────────────────────────────────────────────
echo ">>> Authenticating HuggingFace..."
python -c "from huggingface_hub import login; login('$HF_TOKEN', add_to_git_credential=False)"
if [[ -n "${WANDB_KEY:-}" ]]; then
  echo ">>> Authenticating W&B..."
  python -c "import wandb; wandb.login(key='$WANDB_KEY')"
fi

# ── 5. Verify GPU ────────────────────────────────────────────────────────────
echo ">>> GPU check:"
python -c "
import torch
print('  CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  Device:', torch.cuda.get_device_name(0))
    print('  VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB')
"

# ── 6. Sanity check (tiny-gpt2) ─────────────────────────────────────────────
echo ""
echo ">>> Running MLP-CT sanity check (variant=${VARIANT}, tiny-gpt2)..."
python run.py --config "$SANITY_CONFIG" --mmlu-max-samples 0 --gsm8k-max-samples 0
echo ">>> Sanity check PASSED"

# ── 7. Full training (optional) ──────────────────────────────────────────────
if [[ "$FULL" == true ]]; then
  echo ""
  echo ">>> Starting MLP Consistency Training..."
  echo "    Config:  $FULL_CONFIG"
  echo "    Variant: $VARIANT"
  echo "    Model:   meta-llama/Llama-3.1-8B-Instruct + LoRA"
  echo "    Data:    clear-harm"
  echo ""
  SYCO_DATA="datasets/sycophancy_bct"
  python run.py --config "$FULL_CONFIG" \
    --data-source clear-harm \
    --bct-cot "$SYCO_DATA/bct_cot.jsonl" \
    --bct-noncot "$SYCO_DATA/bct_non_cot.jsonl" \
    --control-cot "$SYCO_DATA/control_cot.jsonl" \
    --control-noncot "$SYCO_DATA/control_non_cot.jsonl"
  echo ">>> Training complete."
else
  echo ""
  echo "Sanity check complete. To start full training, run:"
  echo "  bash runpod_mlp.sh --full --variant $VARIANT"
fi
