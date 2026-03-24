#!/usr/bin/env bash
# ACT sweep stream: Gemma-3-2B-IT (A40)
#
# Runs 9-14:
#    9. act_sycophancy_gemma3_2b_lora          — LoRA q+v LR=5e-6
#   10. act_clearharm_gemma3_2b_lora           — LoRA q+v LR=5e-6
#   11. act_sycophancy_gemma3_2b_fullft_lr1e6  — Full FT LR=1e-6
#   12. act_clearharm_gemma3_2b_fullft_lr1e6   — Full FT LR=1e-6
#   13. act_sycophancy_gemma3_2b_fullft_lr5e7  — Full FT LR=5e-7
#   14. act_clearharm_gemma3_2b_fullft_lr5e7   — Full FT LR=5e-7
#
# Evals per run (pre and post):
#   - MMLU (200 questions)
#   - MT-Bench (80 questions, judged by Gemini Flash)
#   - Persona behavioral prefix k=20
#   - Persona behavioral suffix k=20
#   - ClearHarm behavioral refusal rate
#
# Usage:
#   export HF_HOME=/workspace/hf_cache
#   bash sweep_act_stream_gemma3_2b.sh
#
# Prerequisites:
#   - .env file with OPENROUTER_API_KEY (for MT-Bench and behavioral evals)
#   - tmux recommended for long-running execution

set -euo pipefail

# Source .env for OPENROUTER_API_KEY if available (needed by MT-Bench and behavioral evals)
if [ -f .env ]; then
    set -a
    source .env
    set +a
    echo "Loaded .env"
fi

# ─── Helper: generate a W&B run ID ──────────────────────────────────────────────

new_run_id() {
    python -c "import secrets; print(secrets.token_hex(4))"
}

# ─── Helper: run all evals for a given checkpoint, phase, model, and W&B run ───
#
# Args:
#   $1 — checkpoint path (empty string = base model)
#   $2 — phase ("pre" or "post")
#   $3 — W&B run ID
#   $4 — model name (e.g. "google/gemma-3-2b-it"; empty = default from config.yaml)
#   $5 — W&B run name (only needed on first call to create the run)
#   $6 — W&B group

run_all_evals() {
    local ckpt="$1"
    local phase="$2"
    local run_id="$3"
    local model_name="${4:-}"
    local run_name="${5:-}"
    local wandb_group="${6:-}"

    local ckpt_arg=""
    if [ -n "$ckpt" ]; then ckpt_arg="--checkpoint $ckpt"; fi
    local model_arg=""
    if [ -n "$model_name" ]; then model_arg="--model $model_name"; fi
    local name_args=""
    if [ -n "$run_name" ]; then name_args="--run-name $run_name --wandb-group $wandb_group"; fi

    echo "  [$phase] MMLU..."
    python eval_mmlu.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] MT-Bench..."
    python eval_mtbench.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] Persona behavioral (prefix, k=20)..."
    python eval_persona_behavioral.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --k 20 \
        --facts-position prefix \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] Persona behavioral (suffix, k=20)..."
    python eval_persona_behavioral.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --k 20 \
        --facts-position suffix \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] ClearHarm behavioral refusal..."
    python eval_clearharm_behavioral.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"
}

# ─── Helper: run one full experiment (pre-eval, train, post-eval) ───────────────
#
# Args:
#   $1 — config file
#   $2 — W&B group / experiment name
#   $3 — W&B run name (descriptive)
#   $4 — checkpoint path after training
#   $5 — model name for eval scripts

run_experiment() {
    local config="$1"
    local wandb_group="$2"
    local run_name="$3"
    local ckpt="$4"
    local model_name="${5:-}"

    echo ""
    echo "========================================"
    echo " $wandb_group"
    echo "========================================"
    echo ""

    local run_id
    run_id=$(new_run_id)
    echo "W&B run ID: $run_id"

    # ── Pre-training evals (base model, no checkpoint) ──
    echo "--- Pre-training evals ---"
    run_all_evals "" "pre" "$run_id" "$model_name" "$run_name" "$wandb_group"

    # ── Training ──
    echo "--- Training ---"
    python run.py \
        --config "$config" \
        --run-name "$run_name" \
        --wandb-group "$wandb_group" \
        --wandb-run-id "$run_id" \
        --skip-eval

    # ── Post-training evals (with checkpoint) ──
    echo "--- Post-training evals ---"
    run_all_evals "$ckpt" "post" "$run_id" "$model_name"
}

# ═══════════════════════════════════════════════════════════════════════════════
# GEMMA-3-2B-IT EXPERIMENTS (3 variants x 2 data sources = 6 runs)
# ═══════════════════════════════════════════════════════════════════════════════

GEMMA3_2B_MODEL="google/gemma-3-2b-it"

# ── Gemma-3-2B LoRA q+v LR=5e-6 ──

run_experiment \
    "configs/act_sycophancy_gemma3_2b_lora.yaml" \
    "act_sycophancy_gemma3_2b_lora" \
    "Gemma3-2B_Sycophancy_ACT_LoRA-qv_lr5e-6_w1e-4" \
    "checkpoints/act_sycophancy_gemma3_2b_lora/epoch_1" \
    "$GEMMA3_2B_MODEL"

run_experiment \
    "configs/act_clearharm_gemma3_2b_lora.yaml" \
    "act_clearharm_gemma3_2b_lora" \
    "Gemma3-2B_ClearHarm_ACT_LoRA-qv_lr5e-6_w1e-4" \
    "checkpoints/act_clearharm_gemma3_2b_lora/epoch_1" \
    "$GEMMA3_2B_MODEL"

# ── Gemma-3-2B Full FT LR=1e-6 ──

run_experiment \
    "configs/act_sycophancy_gemma3_2b_fullft_lr1e6.yaml" \
    "act_sycophancy_gemma3_2b_fullft_lr1e6" \
    "Gemma3-2B_Sycophancy_ACT_FullFT_lr1e-6_w1e-4" \
    "checkpoints/act_sycophancy_gemma3_2b_fullft_lr1e6/epoch_1" \
    "$GEMMA3_2B_MODEL"

run_experiment \
    "configs/act_clearharm_gemma3_2b_fullft_lr1e6.yaml" \
    "act_clearharm_gemma3_2b_fullft_lr1e6" \
    "Gemma3-2B_ClearHarm_ACT_FullFT_lr1e-6_w1e-4" \
    "checkpoints/act_clearharm_gemma3_2b_fullft_lr1e6/epoch_1" \
    "$GEMMA3_2B_MODEL"

# ── Gemma-3-2B Full FT LR=5e-7 (Goldilocks) ──

run_experiment \
    "configs/act_sycophancy_gemma3_2b_fullft_lr5e7.yaml" \
    "act_sycophancy_gemma3_2b_fullft_lr5e7" \
    "Gemma3-2B_Sycophancy_ACT_FullFT_lr5e-7_w1e-4" \
    "checkpoints/act_sycophancy_gemma3_2b_fullft_lr5e7/epoch_1" \
    "$GEMMA3_2B_MODEL"

run_experiment \
    "configs/act_clearharm_gemma3_2b_fullft_lr5e7.yaml" \
    "act_clearharm_gemma3_2b_fullft_lr5e7" \
    "Gemma3-2B_ClearHarm_ACT_FullFT_lr5e-7_w1e-4" \
    "checkpoints/act_clearharm_gemma3_2b_fullft_lr5e7/epoch_1" \
    "$GEMMA3_2B_MODEL"

echo ""
echo "========================================"
echo " Gemma-3-2B stream complete (6 runs)."
echo "========================================"
