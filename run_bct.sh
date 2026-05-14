#!/usr/bin/env bash
# BCT (Bias-Augmented Consistency Training / SFT) full pipeline for RunPod.
#
# Mirrors run_act.sh structure: tests → pre-evals → training → post-evals,
# all sharing one W&B run. Eval phase uses run_evals.py — the unified runner
# that loads vLLM ONCE and does sycophancy + clearharm + persona + mtbench
# + MMLU + BRR + rollouts in a single session, with transcripts saved by
# default. Final adapter is async-pushed to HF when --hf-repo is set.
#
# Differences vs. run_act.sh:
#   - Loss is SFTLoss (BCTTrainer path); no consistency pairs.
#   - `--data-mode sycophancy` is required by run.py and passed automatically.
#   - --no-checkpoint suppresses BOTH mid-train evals and mid-train saves.
#
# Required env vars:
#   WANDB_API_KEY       — W&B API key
#   HF_TOKEN            — HuggingFace token (gated models, also for --hf-repo push)
#   OPENROUTER_API_KEY  — OpenRouter key (Gemini judges)
#
# Usage:
#   bash run_bct.sh                                                    # sanity smoke
#   bash run_bct.sh --full                                             # full pipeline (Llama default)
#   bash run_bct.sh --full --config configs/bct_lora_gemma3_4b_lr5e6.yaml
#   bash run_bct.sh --full --hf-repo neilshah/bct-llama31-8b           # push final adapter to HF
#   bash run_bct.sh --full --transcripts-dir /workspace/transcripts/bct-llama
#   bash run_bct.sh --full --skip-pre-evals                            # skip pre-evals (resumed run)
#   bash run_bct.sh --full --skip-rollouts                             # skip multi-turn rollouts

set -euo pipefail

FULL=false
RESUME_RUN_ID=""
SKIP_TRAINING=false
SKIP_PRE_EVALS=false
SKIP_ROLLOUTS=false
TRANSCRIPTS_DIR=""
HF_REPO=""
CONFIG="configs/bct_lora_llama31_8b.yaml"
BCT_ROOT=""
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--full"             ]] && FULL=true
    [[ "${args[$i]}" == "--resume-run-id"    ]] && RESUME_RUN_ID="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--skip-training"    ]] && SKIP_TRAINING=true
    [[ "${args[$i]}" == "--skip-pre-evals"   ]] && SKIP_PRE_EVALS=true
    [[ "${args[$i]}" == "--skip-rollouts"    ]] && SKIP_ROLLOUTS=true
    [[ "${args[$i]}" == "--transcripts-dir"  ]] && TRANSCRIPTS_DIR="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--config"           ]] && CONFIG="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--hf-repo"          ]] && HF_REPO="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--bct-root"         ]] && BCT_ROOT="${args[$((i+1))]:-}"
done

HF_REPO_ARG=""
[[ -n "$HF_REPO" ]] && HF_REPO_ARG="--hf-repo $HF_REPO"

BCT_ROOT_ARG=""
[[ -n "$BCT_ROOT" ]] && BCT_ROOT_ARG="--bct-root $BCT_ROOT"

# Derive identifiers from config
MODEL=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['name'])")
SAVE_DIR=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('save_dir','checkpoints/bct') or 'checkpoints/bct')")
EPOCHS=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('epochs',1))")
QUANTIZATION=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('model',{}).get('quantization','') or '')")
QUANT_ARG=""
[[ -n "$QUANTIZATION" ]] && QUANT_ARG="--quantization $QUANTIZATION"

CONFIG_STEM=$(basename "$CONFIG" .yaml)
SANITY_CONFIG="${CONFIG%.yaml}_sanity.yaml"

# Default a transcripts dir if the user didn't specify one.
if [[ -z "$TRANSCRIPTS_DIR" ]]; then
    TRANSCRIPTS_DIR="results/transcripts/${CONFIG_STEM}"
fi
mkdir -p "$TRANSCRIPTS_DIR/pre" "$TRANSCRIPTS_DIR/post"

TEST_ROOT="${COT_TEST_ROOT:-/workspace/cot-transparency/dataset_dumps/test}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

uv sync --quiet
source .venv/bin/activate

echo "==> Config        : $CONFIG"
echo "==> Model         : $MODEL"
echo "==> SaveDir       : $SAVE_DIR"
echo "==> Epochs        : $EPOCHS"
echo "==> Transcripts   : $TRANSCRIPTS_DIR"
[[ -n "$HF_REPO" ]] && echo "==> HF Repo       : $HF_REPO (final checkpoint pushed asynchronously)"

# ── 0. Install flash-attn (skip if already installed) ─────────────────────────
if python -c "import flash_attn" 2>/dev/null; then
    echo "==> flash-attn already installed, skipping."
else
    echo "==> Installing flash-attn..."
    pip install flash-attn --no-build-isolation -q
fi

# ── 1. Checks ─────────────────────────────────────────────────────────────────
echo "==> Checking environment..."
[[ -z "${WANDB_API_KEY:-}"      ]] && { echo "ERROR: WANDB_API_KEY not set";      exit 1; }
[[ -z "${HF_TOKEN:-}"           ]] && { echo "ERROR: HF_TOKEN not set";           exit 1; }
[[ -z "${OPENROUTER_API_KEY:-}" ]] && { echo "ERROR: OPENROUTER_API_KEY not set"; exit 1; }

python -c "import torch; assert torch.cuda.is_available(), 'No CUDA GPU found'"
echo "    GPU: $(python -c "import torch; print(torch.cuda.get_device_name(0))")"

# ── 2. Login ──────────────────────────────────────────────────────────────────
echo "==> Logging in..."
python -m wandb login "$WANDB_API_KEY"
python -c "from huggingface_hub import login; import os; login(token=os.environ['HF_TOKEN'])"

# ── 3. Tests ──────────────────────────────────────────────────────────────────
python -m pytest tests/test_eval_imports.py losses/test_losses.py data/test_attct_datasets.py data/test_bct_dataset.py -q
echo "    Tests passed."

# ─────────────────────────────────────────────────────────────────────────────
# SANITY MODE
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$FULL" == "false" ]]; then
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    if [[ -f "$SANITY_CONFIG" ]]; then
        echo "==> [SANITY] training run using $SANITY_CONFIG (W&B: $WANDB_RUN_ID)..."
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
            --config "$SANITY_CONFIG" \
            --data-mode sycophancy \
            --no-checkpoint \
            --wandb-run-id "$WANDB_RUN_ID"
    else
        echo "==> [SANITY] No sanity config at $SANITY_CONFIG; running 50-step smoke from $CONFIG."
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
            --config "$CONFIG" \
            --data-mode sycophancy \
            --no-checkpoint \
            --max-steps 50 \
            --wandb-run-id "$WANDB_RUN_ID"
    fi
    unset WANDB_RUN_ID
    echo "Sanity check passed. Re-run with --full for the real pipeline."
    exit 0
fi

# ─────────────────────────────────────────────────────────────────────────────
# FULL PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

if [[ -n "$RESUME_RUN_ID" ]]; then
    export WANDB_RUN_ID="$RESUME_RUN_ID"
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  Resuming W&B run ID: $WANDB_RUN_ID"
    echo "════════════════════════════════════════════════════"
else
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  W&B run ID: $WANDB_RUN_ID"
    echo "════════════════════════════════════════════════════"
fi

# Build rollout flags shared by pre and post (skipped via --skip-rollouts).
ROLLOUT_FLAGS=""
if [[ "$SKIP_ROLLOUTS" == "false" ]]; then
    ROLLOUT_FLAGS="--rollout-tasks frustration,selfdeletion \
        --rollout-datasets \
            wildchat_v3:datasets/wildchat_frustration_v3_test.jsonl:25 \
            math_v3:datasets/math_puzzles_v3_test.jsonl:15 \
        --rollout-n-turns 20"
fi

# ── 4. Pre-training baseline evals ────────────────────────────────────────────
if [[ "$SKIP_PRE_EVALS" == "true" ]]; then
    echo ""
    echo "── PRE-TRAINING EVALS (SKIPPED via --skip-pre-evals) ──"
elif [[ -z "$RESUME_RUN_ID" || "$SKIP_TRAINING" == "true" ]]; then
    echo ""
    echo "── PRE-TRAINING EVALS (single vLLM load) ──────────"

    PRE_BRR_FLAGS="--brr-test-root $TEST_ROOT --brr-limit 300 \
                   --brr-output-json $RESULTS_DIR/pre_brr.json"

    PRE_ROLLOUT_FLAGS="$ROLLOUT_FLAGS"
    [[ -n "$PRE_ROLLOUT_FLAGS" ]] && PRE_ROLLOUT_FLAGS="$PRE_ROLLOUT_FLAGS --rollout-n-samples 3"

    # Pre-eval failures HALT the script — running training without a baseline
    # is worse than waiting to fix the eval. Use --skip-pre-evals to opt out.
    python run_evals.py \
        --model "$MODEL" \
        --metric-prefix "pre/" \
        --wandb-run-id "$WANDB_RUN_ID" \
        --n-syco 200 --n-clearharm 179 --persona-k 10 --persona-n-samples 5 \
        --skip-mtbench \
        --output-root "$TRANSCRIPTS_DIR/pre" \
        --transcripts-dir "$TRANSCRIPTS_DIR/pre" \
        $PRE_BRR_FLAGS \
        $PRE_ROLLOUT_FLAGS \
        $QUANT_ARG
fi

# ── 5. BCT training ───────────────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == "false" ]]; then
    echo ""
    echo "── BCT TRAINING ────────────────────────────────────"
    echo "==> Training with $CONFIG (W&B run: $WANDB_RUN_ID)..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
        --config "$CONFIG" \
        --data-mode sycophancy \
        $BCT_ROOT_ARG \
        --no-checkpoint \
        $HF_REPO_ARG \
        --wandb-run-id "$WANDB_RUN_ID"
else
    echo ""
    echo "── BCT TRAINING (SKIPPED via --skip-training) ──────"
fi

# ── 6. Resolve final checkpoint ───────────────────────────────────────────────
# train.py writes <run_name>__<tag>__<YYYYMMDD_HHMMSS>; pick the most recent.
FINAL_CHECKPOINT=$(ls -dt "$SAVE_DIR"/*"epoch_${EPOCHS}"* 2>/dev/null | head -1 || true)

# Resume-on-fresh-pod fallback: if --skip-training is set and we didn't find a
# local checkpoint but --hf-repo points at one, pull the most recent
# epoch_${EPOCHS} subfolder down from HF Hub into $SAVE_DIR.
if [[ -z "$FINAL_CHECKPOINT" || ! -d "$FINAL_CHECKPOINT" ]]; then
    if [[ "$SKIP_TRAINING" == "true" && -n "$HF_REPO" ]]; then
        echo "==> No local checkpoint found; pulling latest epoch_${EPOCHS} from HF: $HF_REPO"
        mkdir -p "$SAVE_DIR"
        HF_SUBFOLDER=$(uv run --no-project python -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('$HF_REPO')
subs = sorted({f.split('/', 1)[0] for f in files if '/' in f and 'epoch_${EPOCHS}__' in f})
if not subs:
    raise SystemExit('no matching epoch_${EPOCHS}__* subfolder on HF')
print(subs[-1])
")
        echo "==> HF subfolder: $HF_SUBFOLDER"
        uv run --no-project python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='$HF_REPO', allow_patterns='$HF_SUBFOLDER/*', local_dir='$SAVE_DIR')
"
        FINAL_CHECKPOINT="$SAVE_DIR/$HF_SUBFOLDER"
    fi
fi

if [[ -z "$FINAL_CHECKPOINT" || ! -d "$FINAL_CHECKPOINT" ]]; then
    echo "ERROR: No final checkpoint found under $SAVE_DIR matching epoch_${EPOCHS}."
    echo "       Pass --hf-repo <repo> alongside --skip-training to pull from HF."
    exit 1
fi
echo "==> Final checkpoint: $FINAL_CHECKPOINT"

# ── 7. Post-training evals (single vLLM load on the trained adapter) ──────────
echo ""
echo "── POST-TRAINING EVALS (single vLLM load) ─────────"

POST_BRR_FLAGS="--brr-test-root $TEST_ROOT --brr-limit 300 \
                --brr-output-json $RESULTS_DIR/post_brr.json"
[[ -f "$RESULTS_DIR/pre_brr.json" ]] && \
    POST_BRR_FLAGS="$POST_BRR_FLAGS --brr-baseline-json $RESULTS_DIR/pre_brr.json"

POST_ROLLOUT_FLAGS="$ROLLOUT_FLAGS"
[[ -n "$POST_ROLLOUT_FLAGS" ]] && POST_ROLLOUT_FLAGS="$POST_ROLLOUT_FLAGS --rollout-n-samples 3"

python run_evals.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --metric-prefix "post/" \
    --wandb-run-id "$WANDB_RUN_ID" \
    --n-syco 200 --n-clearharm 179 --persona-k 10 --persona-n-samples 5 \
    --n-questions 80 \
    --n-mmlu 1000 \
    --output-root "$TRANSCRIPTS_DIR/post" \
    --transcripts-dir "$TRANSCRIPTS_DIR/post" \
    $POST_BRR_FLAGS \
    $POST_ROLLOUT_FLAGS \
    $QUANT_ARG \
    || echo "WARNING: post-eval run failed (non-fatal)"

unset WANDB_RUN_ID

echo ""
echo "════════════════════════════════════════════════════"
echo "==> Done."
echo "    W&B         : https://wandb.ai/$(python -m wandb whoami 2>/dev/null | head -1)/AttCT"
echo "    Transcripts : $TRANSCRIPTS_DIR"
[[ -n "$HF_REPO" ]] && echo "    HF Hub      : https://huggingface.co/$HF_REPO"
echo "════════════════════════════════════════════════════"
