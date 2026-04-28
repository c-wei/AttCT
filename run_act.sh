#!/usr/bin/env bash
# ACT (Activation Consistency Training) full pipeline for RunPod.
#
# Mirrors run_bct.sh: tests → pre-evals → training → post-evals.
# Differences vs. run_bct.sh:
#   - Loss is ActivationConsistencyLoss (paper Eq. 1, longest matching suffix).
#   - --data-mode sycophancy is required by run.py and passed automatically.
#   - --no-checkpoint suppresses BOTH mid-train evals and mid-train saves
#     (the only useful checkpoint is the final epoch).
#   - Optional --hf-repo username/repo pushes the final adapter to HF Hub
#     asynchronously as it is saved (see train.py:_save_checkpoint).
#
# Required env vars:
#   WANDB_API_KEY       — W&B API key
#   HF_TOKEN            — HuggingFace token (gated models: Llama, Gemma; also for --hf-repo push)
#   OPENROUTER_API_KEY  — OpenRouter key (sycophancy / clearharm / persona / mt-bench Gemini judges)
#
# Usage:
#   bash run_act.sh                                                   # sanity smoke (Llama default)
#   bash run_act.sh --full                                            # full pipeline (Llama default)
#   bash run_act.sh --full --config configs/act_sycophancy_gemma3_4b_v2.yaml
#   bash run_act.sh --full --config configs/act_sycophancy_llama31_8b_v2.yaml --hf-repo username/llama-act
#   bash run_act.sh --full --skip-pre-evals                           # skip pre-evals (resumed run)

set -euo pipefail

FULL=false
RESUME_RUN_ID=""
SKIP_TRAINING=false
SKIP_PRE_EVALS=false
TRANSCRIPTS_DIR=""
HF_REPO=""
CONFIG="configs/act_sycophancy_llama31_8b_v2.yaml"
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--full"             ]] && FULL=true
    [[ "${args[$i]}" == "--resume-run-id"    ]] && RESUME_RUN_ID="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--skip-training"    ]] && SKIP_TRAINING=true
    [[ "${args[$i]}" == "--skip-pre-evals"   ]] && SKIP_PRE_EVALS=true
    [[ "${args[$i]}" == "--transcripts-dir"  ]] && TRANSCRIPTS_DIR="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--config"           ]] && CONFIG="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--hf-repo"          ]] && HF_REPO="${args[$((i+1))]:-}"
done

TRANSCRIPTS_ARG=""
[[ -n "$TRANSCRIPTS_DIR" ]] && TRANSCRIPTS_ARG="--transcripts-dir $TRANSCRIPTS_DIR"

HF_REPO_ARG=""
[[ -n "$HF_REPO" ]] && HF_REPO_ARG="--hf-repo $HF_REPO"

# Derive model name, save dir, epoch count, and final checkpoint path from config
MODEL=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c['model']['name'])")
SAVE_DIR=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('save_dir','checkpoints/act') or 'checkpoints/act')")
EPOCHS=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('training',{}).get('epochs',1))")
QUANTIZATION=$(uv run --no-project python -c \
    "import yaml; c=yaml.safe_load(open('$CONFIG')); print(c.get('model',{}).get('quantization','') or '')")
QUANT_ARG=""
[[ -n "$QUANTIZATION" ]] && QUANT_ARG="--quantization $QUANTIZATION"

# train.py timestamps checkpoint dirs as <run_name>__<tag>__<YYYYMMDD_HHMMSS>,
# so we can't hard-code the final path; we discover it after training.
# For resumed runs the user can pass an explicit --skip-training and we'll just
# pick the most recent epoch_* under SAVE_DIR.
SANITY_CONFIG="${CONFIG%.yaml}_sanity.yaml"

TEST_ROOT="${COT_TEST_ROOT:-/workspace/cot-transparency/dataset_dumps/test}"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

uv sync --quiet
source .venv/bin/activate

echo "==> Config : $CONFIG"
echo "==> Model  : $MODEL"
echo "==> SaveDir: $SAVE_DIR"
echo "==> Epochs : $EPOCHS"
[[ -n "$HF_REPO" ]] && echo "==> HF Repo: $HF_REPO (final checkpoint will be pushed asynchronously)"

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
python -m pytest losses/test_losses.py data/test_attct_datasets.py -q
echo "    Tests passed."

# ─────────────────────────────────────────────────────────────────────────────
# SANITY MODE
# ─────────────────────────────────────────────────────────────────────────────
if [[ "$FULL" == "false" ]]; then
    if [[ ! -f "$SANITY_CONFIG" ]]; then
        echo "==> [SANITY] No sanity config found at $SANITY_CONFIG"
        echo "    Running 50-step smoke from $CONFIG with --max-steps 50 instead."
        export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
            --config "$CONFIG" \
            --data-mode sycophancy \
            --no-checkpoint \
            --skip-eval \
            --max-steps 50 \
            --wandb-run-id "$WANDB_RUN_ID"
        unset WANDB_RUN_ID
        echo "Sanity check passed. Re-run with --full for the real pipeline."
        exit 0
    fi
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    echo "==> [SANITY] training run using $SANITY_CONFIG (W&B: $WANDB_RUN_ID)..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
        --config "$SANITY_CONFIG" \
        --data-mode sycophancy \
        --no-checkpoint \
        --skip-eval
    unset WANDB_RUN_ID
    echo "Sanity check passed. Re-run with --full for the real pipeline."
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
    if [[ "$SKIP_TRAINING" == "true" ]]; then
        echo "  (--skip-training set: pre-evals WILL run)"
    else
        echo "  (skipping pre-training evals)"
    fi
    echo "════════════════════════════════════════════════════"
else
    export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
    echo ""
    echo "════════════════════════════════════════════════════"
    echo "  W&B run ID: $WANDB_RUN_ID"
    echo "════════════════════════════════════════════════════"
fi

run_eval() {
    local label="$1"; shift
    echo "==> $label..."
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "[dry-run] python $*"
        return 0
    fi
    python "$@" || echo "WARNING: $label failed (non-fatal)"
}

# ── 4. Pre-training baseline evals ────────────────────────────────────────────
if [[ "$SKIP_PRE_EVALS" == "true" ]]; then
    echo ""
    echo "── PRE-TRAINING EVALS (SKIPPED via --skip-pre-evals) ──"
elif [[ -z "$RESUME_RUN_ID" || "$SKIP_TRAINING" == "true" ]]; then
    echo ""
    echo "── PRE-TRAINING EVALS ──────────────────────────────"

    PRE_TRANSCRIPTS_ARG=""
    PRE_ROLLOUT_ARG=""
    if [[ -n "$TRANSCRIPTS_DIR" ]]; then
        PRE_TRANSCRIPTS_ARG="--transcripts-dir $TRANSCRIPTS_DIR/pre"
        PRE_ROLLOUT_ARG="--output-root $TRANSCRIPTS_DIR/pre"
    fi

    run_eval "Pre: BRR eval (base model)" evaluate_bct.py \
        --model "$MODEL" \
        --test_root "$TEST_ROOT" \
        --output_json "$RESULTS_DIR/pre_brr.json" \
        --metric-prefix "pre/" \
        --limit 300 \
        $QUANT_ARG

    run_eval "Pre: all evals" run_evals.py \
        --model "$MODEL" \
        --n-syco 200 --n-clearharm 179 --persona-k 10 --persona-n-samples 5 \
        --skip-mtbench \
        $QUANT_ARG \
        $PRE_TRANSCRIPTS_ARG \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"

    run_eval "Pre: rollout evals (frustration + selfdeletion, wildchat + math)" eval_rollout.py \
        --model "$MODEL" \
        --tasks frustration,selfdeletion \
        --datasets \
            wildchat_v3:datasets/wildchat_frustration_v3_test.jsonl:25 \
            math_v3:datasets/math_puzzles_v3_test.jsonl:15 \
        --n-samples 3 --n-turns 20 \
        $PRE_ROLLOUT_ARG \
        --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "pre/"

fi

# ── 5. ACT training ───────────────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == "false" ]]; then
    echo ""
    echo "── ACT TRAINING ────────────────────────────────────"
    echo "==> Training with $CONFIG (W&B run: $WANDB_RUN_ID)..."
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
        --config "$CONFIG" \
        --data-mode sycophancy \
        --no-checkpoint \
        --skip-eval \
        $HF_REPO_ARG \
        --wandb-run-id "$WANDB_RUN_ID"
else
    echo ""
    echo "── ACT TRAINING (SKIPPED via --skip-training) ──────"
fi

# ── 6. Resolve final checkpoint ───────────────────────────────────────────────
# train.py writes checkpoints as <run_name>__epoch_<N>__<timestamp>; pick the
# most recent epoch_$EPOCHS dir under $SAVE_DIR.
FINAL_CHECKPOINT=$(ls -dt "$SAVE_DIR"/*"epoch_${EPOCHS}"* 2>/dev/null | head -1 || true)
if [[ -z "$FINAL_CHECKPOINT" || ! -d "$FINAL_CHECKPOINT" ]]; then
    echo "ERROR: No final checkpoint found under $SAVE_DIR matching epoch_${EPOCHS}."
    echo "       Training likely crashed before completing all $EPOCHS epoch(s)."
    exit 1
fi
echo "==> Final checkpoint: $FINAL_CHECKPOINT"

# ── 7. Post-training evals ────────────────────────────────────────────────────
echo ""
echo "── POST-TRAINING EVALS ─────────────────────────────"

POST_TRANSCRIPTS_ARG=""
POST_ROLLOUT_ARG=""
POST_BRR_BASELINE_ARG=""
[[ -f "$RESULTS_DIR/pre_brr.json" ]] && POST_BRR_BASELINE_ARG="--baseline_json $RESULTS_DIR/pre_brr.json"
if [[ -n "$TRANSCRIPTS_DIR" ]]; then
    POST_TRANSCRIPTS_ARG="--transcripts-dir $TRANSCRIPTS_DIR/post"
    POST_ROLLOUT_ARG="--output-root $TRANSCRIPTS_DIR/post"
fi

run_eval "Post: all evals" run_evals.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --n-syco 200 --n-clearharm 179 --persona-k 10 --persona-n-samples 5 \
    --n-questions 80 \
    $QUANT_ARG \
    $POST_TRANSCRIPTS_ARG \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

run_eval "Post: MMLU (n=1000)" eval_mmlu.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --n-samples 1000 \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

run_eval "Post: BRR eval" evaluate_bct.py \
    --model "$MODEL" \
    --lora_path "$FINAL_CHECKPOINT" \
    --test_root "$TEST_ROOT" \
    $POST_BRR_BASELINE_ARG \
    --output_json "$RESULTS_DIR/post_brr.json" \
    --metric-prefix "post/" \
    --limit 300 \
    $QUANT_ARG

run_eval "Post: rollout evals (frustration + selfdeletion, wildchat + math)" eval_rollout.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --tasks frustration,selfdeletion \
    --datasets \
        wildchat_v3:datasets/wildchat_frustration_v3_test.jsonl:25 \
        math_v3:datasets/math_puzzles_v3_test.jsonl:15 \
    --n-samples 5 --n-turns 20 \
    $POST_ROLLOUT_ARG \
    --wandb-run-id "$WANDB_RUN_ID" --metric-prefix "post/"

unset WANDB_RUN_ID

echo ""
echo "════════════════════════════════════════════════════"
echo "==> Done."
echo "    W&B : https://wandb.ai/$(python -m wandb whoami 2>/dev/null | head -1)/AttCT"
[[ -n "$HF_REPO" ]] && echo "    HF  : https://huggingface.co/$HF_REPO"
echo "════════════════════════════════════════════════════"
