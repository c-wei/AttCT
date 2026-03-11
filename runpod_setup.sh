#!/usr/bin/env bash
# RunPod setup and training script for ACT sycophancy replication.
# Run this after SSHing into a RunPod instance (A40 48GB or A100 80GB).
#
# STORAGE: Attach a network volume (>=30 GB) mounted at /workspace BEFORE
# starting the pod. The script stores the HuggingFace model cache and
# checkpoints on /workspace so they persist across pod restarts.
#   - Llama-3.1-8B weights:  ~16 GB  (downloaded once, cached)
#   - MMLU dataset cache:    ~500 MB
#   - LoRA checkpoints:      ~10 MB/epoch
#   → 30 GB volume recommended
#
# Usage:
#   bash runpod_setup.sh            # setup + sanity check only
#   bash runpod_setup.sh --full     # setup + sanity check + full training
#
# Required environment variables (set before running, or export inline):
#   HF_TOKEN   — HuggingFace token with access to meta-llama/Llama-3.1-8B
#   WANDB_KEY  — Weights & Biases API key
#
# Example:
#   HF_TOKEN=hf_xxx WANDB_KEY=xxx bash runpod_setup.sh --full

set -euo pipefail

REPO_URL="https://github.com/c-wei/AttCT.git"
BRANCH="replicate_act_consistency"
WORKDIR="/workspace/AttCT-act-consistency"

# Redirect HuggingFace cache to network volume (avoids filling container disk)
export HF_HOME="/workspace/hf_cache"
export HF_DATASETS_CACHE="/workspace/hf_cache/datasets"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE"

# ── 1. Check required env vars ───────────────────────────────────────────────
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. Export it before running this script."
  exit 1
fi
if [[ -z "${WANDB_KEY:-}" ]]; then
  echo "ERROR: WANDB_KEY is not set. Export it before running this script."
  exit 1
fi

# Check network volume is actually mounted (not just container disk)
WORKSPACE_AVAIL_GB=$(df -BG /workspace | awk 'NR==2 {gsub("G",""); print $4}')
if [[ "$WORKSPACE_AVAIL_GB" -lt 20 ]]; then
  echo "WARNING: /workspace only has ${WORKSPACE_AVAIL_GB}GB free."
  echo "  Llama-3.1-8B requires ~16GB. Attach a >=30GB network volume at /workspace."
  echo "  Continuing anyway — will fail if disk fills up."
fi
echo ">>> /workspace available: ${WORKSPACE_AVAIL_GB}GB"

# ── 2. Clone repo ────────────────────────────────────────────────────────────
if [[ ! -d "$WORKDIR" ]]; then
  echo ">>> Cloning repo (branch: $BRANCH)..."
  git clone --branch "$BRANCH" "$REPO_URL" "$WORKDIR"
else
  echo ">>> Repo already exists at $WORKDIR, pulling latest..."
  git -C "$WORKDIR" pull
fi

cd "$WORKDIR"

# ── 3. Install dependencies ──────────────────────────────────────────────────
echo ">>> Installing Python dependencies..."
pip install --quiet torch transformers peft datasets wandb tqdm pyyaml

# ── 4. Authenticate ──────────────────────────────────────────────────────────
echo ">>> Authenticating HuggingFace and W&B..."
python -c "from huggingface_hub import login; login('$HF_TOKEN', add_to_git_credential=False)"
python -c "import wandb; wandb.login(key='$WANDB_KEY')"

# ── 5. Verify GPU ────────────────────────────────────────────────────────────
echo ">>> GPU check:"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); print('  Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print('  VRAM:', round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), 'GB' if torch.cuda.is_available() else '')"

# ── 6. Sanity check (fast, ~2 min on A100) ───────────────────────────────────
echo ""
echo ">>> Running GPU sanity check (20 prompts, 1 epoch)..."
python run.py --config configs/sanity_act_gpu.yaml
echo ">>> Sanity check PASSED"

# ── 7. Full training run (optional) ──────────────────────────────────────────
if [[ "${1:-}" == "--full" ]]; then
  echo ""
  echo ">>> Starting full ACT sycophancy training..."
  echo "    Config: configs/act_sycophancy.yaml"
  echo "    Data:   datasets/sycophancy_bct/control_cot.jsonl (5000 prompts)"
  echo "    Model:  meta-llama/Llama-3.1-8B + LoRA (r=8)"
  echo ""
  python run.py --config configs/act_sycophancy.yaml
  echo ">>> Full training complete."
else
  echo ""
  echo "Sanity check complete. To start full training, run:"
  echo "  cd $WORKDIR && python run.py --config configs/act_sycophancy.yaml"
fi
