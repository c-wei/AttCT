#!/usr/bin/env bash
# RunPod setup and training script for ACT sycophancy replication.
# Run this after SSHing into a RunPod instance (A40 48GB or A100 80GB).
#
# STORAGE: Attach a network volume (>=30 GB) mounted at /workspace BEFORE
# starting the pod. The script stores the HuggingFace model cache and
# checkpoints on /workspace so they persist across pod restarts.
#   - Llama-3.1-8B weights:  ~16 GB  (downloaded once, cached)
#   - Gemma 2 2B weights:    ~5 GB
#   - MMLU dataset cache:    ~500 MB
#   - LoRA checkpoints:      ~10 MB/epoch
#   → 30 GB volume recommended
#
# Usage:
#   bash runpod_setup.sh --model gemma2b           # sanity check only
#   bash runpod_setup.sh --model gemma2b --full    # sanity + full training
#   bash runpod_setup.sh --model llama --full      # Llama-3.1-8B full run
#
# Models:
#   gemma2b  → google/gemma-2-2b   (default, matches paper)
#   llama    → meta-llama/Llama-3.1-8B
#
# Required environment variables:
#   HF_TOKEN   — HuggingFace token (needs access to the chosen model)
#   WANDB_KEY  — Weights & Biases API key

set -euo pipefail

# ── Parse arguments ───────────────────────────────────────────────────────────
MODEL="gemma2b"
FULL=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2 ;;
    --full)  FULL=true; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

# Map model name to configs
case "$MODEL" in
  gemma2b)
    SANITY_CONFIG="configs/sanity_act_gpu_gemma2b.yaml"
    FULL_CONFIG="configs/act_sycophancy_gemma2b.yaml"
    MODEL_LABEL="google/gemma-2-2b"
    ;;
  llama)
    SANITY_CONFIG="configs/sanity_act_gpu.yaml"
    FULL_CONFIG="configs/act_sycophancy.yaml"
    MODEL_LABEL="meta-llama/Llama-3.1-8B"
    ;;
  *)
    echo "ERROR: Unknown model '$MODEL'. Choose: gemma2b, llama"
    exit 1
    ;;
esac

REPO_URL="https://github.com/<anon>/<repo>.git"
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
  echo "  Model download requires ~16GB. Attach a >=30GB network volume at /workspace."
  echo "  Continuing anyway — will fail if disk fills up."
fi
echo ">>> /workspace available: ${WORKSPACE_AVAIL_GB}GB"
echo ">>> Model: $MODEL_LABEL"

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

# ── 6. Sanity check ───────────────────────────────────────────────────────────
echo ""
echo ">>> Running GPU sanity check (20 prompts, 1 epoch)..."
python run.py --config "$SANITY_CONFIG"
echo ">>> Sanity check PASSED"

# ── 7. Full training run (optional) ──────────────────────────────────────────
if [[ "$FULL" == true ]]; then
  echo ""
  echo ">>> Starting full ACT sycophancy training..."
  echo "    Config: $FULL_CONFIG"
  echo "    Data:   datasets/sycophancy_bct/control_cot.jsonl (5000 prompts)"
  echo "    Model:  $MODEL_LABEL + LoRA (r=8)"
  echo ""
  python run.py --config "$FULL_CONFIG"
  echo ">>> Full training complete."
else
  echo ""
  echo "Sanity check complete. To start full training, run:"
  echo "  cd $WORKDIR && python run.py --config $FULL_CONFIG"
fi
