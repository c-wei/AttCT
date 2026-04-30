#!/usr/bin/env bash
# ACT (Activation Consistency Training) full pipeline for RunPod.
#
# Pre-evals → ACT training → post-evals, all sharing one W&B run.
#
# Eval phase uses run_evals.py — the unified runner that loads vLLM ONCE and
# does sycophancy + clearharm + persona + mtbench + MMLU + BRR + rollouts in
# a single session (about 4× faster than run_bct.sh's per-eval subprocess
# pattern, which is the older approach). Transcripts (per-record prompt /
# response / verdict for syco / clearharm / persona) are saved by default.
#
# Required env vars:
#   WANDB_API_KEY       — W&B API key
#   HF_TOKEN            — HuggingFace token (gated models, also for --hf-repo)
#   OPENROUTER_API_KEY  — OpenRouter key (Gemini judges)
#
# Usage:
#   bash run_act.sh                                                   # sanity smoke
#   bash run_act.sh --full                                            # full pipeline (Llama default)
#   bash run_act.sh --full --config configs/act_sycophancy_gemma3_4b_v2.yaml
#   bash run_act.sh --full --hf-repo username/llama-act               # push final adapter to HF
#   bash run_act.sh --full --transcripts-dir /workspace/transcripts/llama
#   bash run_act.sh --full --skip-pre-evals                           # post-only on a resumed run
#   bash run_act.sh --full --skip-rollouts                            # skip multi-turn rollout evals
#   bash run_act.sh --full --interleave                               # pair each AttCT step with KL-reg on alpaca
#   bash run_act.sh --full --interleave --kl-dataset ultrachat        # alternative KL dataset
#   bash run_act.sh --full --interleave --kl-ratio 0.5                # KL fires on ~50% of AttCT steps

set -euo pipefail

FULL=false
RESUME_RUN_ID=""
SKIP_TRAINING=false
SKIP_PRE_EVALS=false
SKIP_ROLLOUTS=false
TRANSCRIPTS_DIR=""
HF_REPO=""
INTERLEAVE=false
KL_DATASET="alpaca"
KL_RATIO=""
KL_WEIGHT=""
ROLLOUT_N_SAMPLES="3"
N_ANTHROPIC="0"
ANTHROPIC_ONLY=false
SKIP_SYCOPHANCY=false
SKIP_CLEARHARM=false
SKIP_PERSONA=false
SKIP_MTBENCH=false
SKIP_MMLU=false
SKIP_BRR=false
CONFIG="configs/act_sycophancy_llama31_8b_v2.yaml"
args=("$@")
for i in "${!args[@]}"; do
    [[ "${args[$i]}" == "--full"             ]] && FULL=true
    [[ "${args[$i]}" == "--resume-run-id"    ]] && RESUME_RUN_ID="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--skip-training"    ]] && SKIP_TRAINING=true
    [[ "${args[$i]}" == "--skip-pre-evals"   ]] && SKIP_PRE_EVALS=true
    [[ "${args[$i]}" == "--skip-rollouts"    ]] && SKIP_ROLLOUTS=true
    [[ "${args[$i]}" == "--skip-sycophancy"  ]] && SKIP_SYCOPHANCY=true
    [[ "${args[$i]}" == "--skip-clearharm"   ]] && SKIP_CLEARHARM=true
    [[ "${args[$i]}" == "--skip-persona"     ]] && SKIP_PERSONA=true
    [[ "${args[$i]}" == "--skip-mtbench"     ]] && SKIP_MTBENCH=true
    [[ "${args[$i]}" == "--skip-mmlu"        ]] && SKIP_MMLU=true
    [[ "${args[$i]}" == "--skip-brr"         ]] && SKIP_BRR=true
    [[ "${args[$i]}" == "--transcripts-dir"  ]] && TRANSCRIPTS_DIR="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--config"           ]] && CONFIG="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--hf-repo"          ]] && HF_REPO="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--interleave"       ]] && INTERLEAVE=true
    [[ "${args[$i]}" == "--kl-dataset"       ]] && KL_DATASET="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--kl-ratio"         ]] && KL_RATIO="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--kl-weight"        ]] && KL_WEIGHT="${args[$((i+1))]:-}"
    [[ "${args[$i]}" == "--rollout-n-samples" ]] && ROLLOUT_N_SAMPLES="${args[$((i+1))]:-3}"
    [[ "${args[$i]}" == "--n-anthropic"      ]] && N_ANTHROPIC="${args[$((i+1))]:-0}"
    [[ "${args[$i]}" == "--anthropic-only"   ]] && ANTHROPIC_ONLY=true
done

# Build skip-flags string forwarded to run_evals.py
SKIP_EVAL_ARGS=""
[[ "$SKIP_SYCOPHANCY" == "true" ]] && SKIP_EVAL_ARGS="$SKIP_EVAL_ARGS --skip-sycophancy"
[[ "$SKIP_CLEARHARM"  == "true" ]] && SKIP_EVAL_ARGS="$SKIP_EVAL_ARGS --skip-clearharm"
[[ "$SKIP_PERSONA"    == "true" ]] && SKIP_EVAL_ARGS="$SKIP_EVAL_ARGS --skip-persona"
[[ "$SKIP_MTBENCH"    == "true" ]] && SKIP_EVAL_ARGS="$SKIP_EVAL_ARGS --skip-mtbench"

# --anthropic-only: post-eval phase runs ONLY the Anthropic sycophancy eval.
# Skips sycophancy MCQ, clearharm, persona, mtbench, MMLU, rollouts, BRR.
# Useful for backfilling Anthropic numbers on existing checkpoints without
# re-running the rest of the eval suite (~50 min vs ~80 min per run).
# Auto-implies --skip-pre-evals and --skip-rollouts; sets --n-anthropic 999
# unless overridden by an explicit --n-anthropic.
ANTHROPIC_ONLY_ARGS=""
if [[ "$ANTHROPIC_ONLY" == "true" ]]; then
    SKIP_PRE_EVALS=true
    SKIP_ROLLOUTS=true
    [[ "$N_ANTHROPIC" -eq 0 ]] && N_ANTHROPIC="999"
    ANTHROPIC_ONLY_ARGS="--skip-sycophancy --skip-clearharm --skip-persona --skip-mtbench"
fi

# Anthropic sycophancy eval (Anthropic/model-written-evals) — disabled by default,
# pass --n-anthropic <N> (e.g. 999) to enable. Only runs in the POST-eval phase.
ANTHROPIC_ARG=""
[[ "$N_ANTHROPIC" -gt 0 ]] && ANTHROPIC_ARG="--n-anthropic $N_ANTHROPIC"

HF_REPO_ARG=""
[[ -n "$HF_REPO" ]] && HF_REPO_ARG="--hf-repo $HF_REPO"

# --interleave passthrough: pairs each AttCT step with a KL regularization step
# on instruct data (default: alpaca) so the model retains base-model behaviour
# on prompts outside the consistency-training distribution.
INTERLEAVE_ARGS=""
if [[ "$INTERLEAVE" == "true" ]]; then
    INTERLEAVE_ARGS="--interleave --kl-dataset $KL_DATASET"
    [[ -n "$KL_RATIO"  ]] && INTERLEAVE_ARGS="$INTERLEAVE_ARGS --kl-ratio $KL_RATIO"
    [[ -n "$KL_WEIGHT" ]] && INTERLEAVE_ARGS="$INTERLEAVE_ARGS --kl-weight $KL_WEIGHT"
fi

# Derive identifiers from config
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

CONFIG_STEM=$(basename "$CONFIG" .yaml)
SANITY_CONFIG="${CONFIG%.yaml}_sanity.yaml"

# Default a transcripts dir if the user didn't specify one. Transcripts include
# per-record prompt/response/verdict for syco/clearharm/persona evals.
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
python -m pytest losses/test_losses.py data/test_attct_datasets.py -q
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
            --skip-eval \
            $INTERLEAVE_ARGS \
            --wandb-run-id "$WANDB_RUN_ID"
    else
        echo "==> [SANITY] No sanity config at $SANITY_CONFIG; running 50-step smoke from $CONFIG."
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
            --config "$CONFIG" \
            --data-mode sycophancy \
            --no-checkpoint \
            --skip-eval \
            --max-steps 50 \
            $INTERLEAVE_ARGS \
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
    [[ -n "$PRE_ROLLOUT_FLAGS" ]] && PRE_ROLLOUT_FLAGS="$PRE_ROLLOUT_FLAGS --rollout-n-samples $ROLLOUT_N_SAMPLES"

    # Pre-eval failures HALT the script — running training without a baseline
    # is worse than waiting to fix the eval. Use --skip-pre-evals if you need
    # to bypass for a known reason. set -euo pipefail at the top of the script
    # propagates a non-zero exit from run_evals.py up to here.
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

# ── 5. ACT training ───────────────────────────────────────────────────────────
if [[ "$SKIP_TRAINING" == "false" ]]; then
    echo ""
    echo "── ACT TRAINING ────────────────────────────────────"
    echo "==> Training with $CONFIG (W&B run: $WANDB_RUN_ID)..."
    [[ "$INTERLEAVE" == "true" ]] && echo "==> Interleaved KL regularization: $KL_DATASET (ratio=${KL_RATIO:-default}, weight=${KL_WEIGHT:-default})"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python run.py \
        --config "$CONFIG" \
        --data-mode sycophancy \
        --no-checkpoint \
        --skip-eval \
        $HF_REPO_ARG \
        $INTERLEAVE_ARGS \
        --wandb-run-id "$WANDB_RUN_ID"
else
    echo ""
    echo "── ACT TRAINING (SKIPPED via --skip-training) ──────"
fi

# ── 6. Resolve final checkpoint ───────────────────────────────────────────────
# train.py writes <run_name>__<tag>__<YYYYMMDD_HHMMSS>; pick the most recent.
FINAL_CHECKPOINT=$(ls -dt "$SAVE_DIR"/*"epoch_${EPOCHS}"* 2>/dev/null | head -1 || true)

# Fallback: some trainers (older InterleavedTrainer commits) only saved step_*
# tags — pick the latest step_* if no epoch_* match. Belt-and-suspenders so the
# chain doesn't abort on a checkpoint-naming mismatch.
if [[ -z "$FINAL_CHECKPOINT" || ! -d "$FINAL_CHECKPOINT" ]]; then
    LOCAL_STEP_CKPT=$(ls -dt "$SAVE_DIR"/*step_* 2>/dev/null | head -1 || true)
    if [[ -n "$LOCAL_STEP_CKPT" && -d "$LOCAL_STEP_CKPT" ]]; then
        echo "==> No epoch_${EPOCHS} checkpoint, falling back to latest step checkpoint: $LOCAL_STEP_CKPT"
        FINAL_CHECKPOINT="$LOCAL_STEP_CKPT"
    fi
fi

# Resume-on-fresh-pod fallback: if --skip-training is set and we didn't find a
# local checkpoint but --hf-repo points at one, pull the most recent
# epoch_${EPOCHS} (or step_*) subfolder down from HF Hub into $SAVE_DIR.
if [[ -z "$FINAL_CHECKPOINT" || ! -d "$FINAL_CHECKPOINT" ]]; then
    if [[ "$SKIP_TRAINING" == "true" && -n "$HF_REPO" ]]; then
        echo "==> No local checkpoint found; pulling latest from HF: $HF_REPO"
        mkdir -p "$SAVE_DIR"
        HF_SUBFOLDER=$(uv run --no-project python -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('$HF_REPO')
# Prefer epoch_${EPOCHS} subfolders; fall back to step_* if absent.
subs_epoch = sorted({f.split('/', 1)[0] for f in files if '/' in f and 'epoch_${EPOCHS}__' in f})
subs_step  = sorted({f.split('/', 1)[0] for f in files if '/' in f and '__step_' in f})
subs = subs_epoch or subs_step
if not subs:
    raise SystemExit('no matching epoch_${EPOCHS}__* or step_* subfolder on HF')
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
[[ -n "$POST_ROLLOUT_FLAGS" ]] && POST_ROLLOUT_FLAGS="$POST_ROLLOUT_FLAGS --rollout-n-samples $ROLLOUT_N_SAMPLES"

# Compose the post-eval flag set. Each can be individually skipped via the
# corresponding --skip-* flag; --anthropic-only is a convenience that skips
# everything except Anthropic.
if [[ "$ANTHROPIC_ONLY" == "true" ]]; then
    POST_BRR_FLAGS=""
    POST_ROLLOUT_FLAGS=""
    POST_MMLU_FLAG="--n-mmlu 0"
else
    POST_MMLU_FLAG="--n-mmlu 1000"
    [[ "$SKIP_MMLU" == "true" ]] && POST_MMLU_FLAG="--n-mmlu 0"
    [[ "$SKIP_BRR"  == "true" ]] && POST_BRR_FLAGS=""
fi

python run_evals.py \
    --model "$MODEL" \
    --checkpoint "$FINAL_CHECKPOINT" \
    --metric-prefix "post/" \
    --wandb-run-id "$WANDB_RUN_ID" \
    --n-syco 200 --n-clearharm 179 --persona-k 10 --persona-n-samples 5 \
    --n-questions 80 \
    $POST_MMLU_FLAG \
    --output-root "$TRANSCRIPTS_DIR/post" \
    --transcripts-dir "$TRANSCRIPTS_DIR/post" \
    $ANTHROPIC_ARG \
    $ANTHROPIC_ONLY_ARGS \
    $SKIP_EVAL_ARGS \
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
