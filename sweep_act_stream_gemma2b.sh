#!/usr/bin/env bash
# ACT sweep stream: Gemma-2-2B-IT all variants (A40)
#
# Runs (6 total):
#   1. act_sycophancy_gemma_lora          — LoRA q+v LR=5e-6
#   2. act_clearharm_gemma_lora           — LoRA q+v LR=5e-6
#   3. act_sycophancy_gemma_fullft_lr1e6  — Full FT LR=1e-6
#   4. act_clearharm_gemma_fullft_lr1e6   — Full FT LR=1e-6
#   5. act_sycophancy_gemma_fullft_lr5e7  — Full FT LR=5e-7 (Goldilocks)
#   6. act_clearharm_gemma_fullft_lr5e7   — Full FT LR=5e-7
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
#   bash sweep_act_stream_gemma2b.sh
#
# Prerequisites:
#   - .env file with OPENROUTER_API_KEY (for MT-Bench and behavioral evals)
#   - tmux recommended for long-running execution (~15h estimated)

set -euo pipefail

if [ -f .env ]; then
    set -a; source .env; set +a
    echo "Loaded .env"
fi

new_run_id() { python -c "import secrets; print(secrets.token_hex(4))"; }

run_all_evals() {
    local ckpt="$1" phase="$2" run_id="$3" model_name="${4:-}" run_name="${5:-}" wandb_group="${6:-}"
    local ckpt_arg=""; [ -n "$ckpt" ] && ckpt_arg="--checkpoint $ckpt"
    local model_arg=""; [ -n "$model_name" ] && model_arg="--model $model_name"
    local name_args=""; [ -n "$run_name" ] && name_args="--run-name $run_name --wandb-group $wandb_group"

    echo "  [$phase] MMLU..."
    python eval_mmlu.py $ckpt_arg $model_arg $name_args --wandb-run-id "$run_id" --metric-prefix "${phase}/"

    echo "  [$phase] MT-Bench..."
    python eval_mtbench.py $ckpt_arg $model_arg $name_args --wandb-run-id "$run_id" --metric-prefix "${phase}/"

    echo "  [$phase] Persona behavioral (prefix, k=20)..."
    python eval_persona_behavioral.py $ckpt_arg $model_arg $name_args --k 20 --facts-position prefix --wandb-run-id "$run_id" --metric-prefix "${phase}/"

    echo "  [$phase] Persona behavioral (suffix, k=20)..."
    python eval_persona_behavioral.py $ckpt_arg $model_arg $name_args --k 20 --facts-position suffix --wandb-run-id "$run_id" --metric-prefix "${phase}/"

    echo "  [$phase] ClearHarm behavioral refusal..."
    python eval_clearharm_behavioral.py $ckpt_arg $model_arg $name_args --wandb-run-id "$run_id" --metric-prefix "${phase}/"
}

run_experiment() {
    local config="$1" wandb_group="$2" run_name="$3" ckpt="$4" model_name="${5:-}"
    echo ""; echo "========================================"; echo " $wandb_group"; echo "========================================"
    local run_id; run_id=$(new_run_id)
    echo "W&B run ID: $run_id"
    echo "--- Pre-training evals ---"
    run_all_evals "" "pre" "$run_id" "$model_name" "$run_name" "$wandb_group"
    echo "--- Training ---"
    python run.py --config "$config" --run-name "$run_name" --wandb-group "$wandb_group" --wandb-run-id "$run_id" --skip-eval
    echo "--- Post-training evals ---"
    run_all_evals "$ckpt" "post" "$run_id" "$model_name"
}

# ═══════════════════════════════════════════════════════════════════════════════
# GEMMA-2-2B-IT — all 6 runs
# ═══════════════════════════════════════════════════════════════════════════════

GEMMA_MODEL="google/gemma-2-2b-it"

# ── LoRA q+v LR=5e-6 (2 runs) ──

run_experiment \
    "configs/act_sycophancy_gemma_lora.yaml" \
    "act_sycophancy_gemma_lora" \
    "Gemma-2B_Sycophancy_ACT_LoRA-qv_lr5e-6_w1e-4" \
    "checkpoints/act_sycophancy_gemma_lora/epoch_1" \
    "$GEMMA_MODEL"

run_experiment \
    "configs/act_clearharm_gemma_lora.yaml" \
    "act_clearharm_gemma_lora" \
    "Gemma-2B_ClearHarm_ACT_LoRA-qv_lr5e-6_w1e-4" \
    "checkpoints/act_clearharm_gemma_lora/epoch_1" \
    "$GEMMA_MODEL"

# ── Full FT LR=1e-6 (2 runs) ──

run_experiment \
    "configs/act_sycophancy_gemma_fullft_lr1e6.yaml" \
    "act_sycophancy_gemma_fullft_lr1e6" \
    "Gemma-2B_Sycophancy_ACT_FullFT_lr1e-6_w1e-4" \
    "checkpoints/act_sycophancy_gemma_fullft_lr1e6/epoch_1" \
    "$GEMMA_MODEL"

run_experiment \
    "configs/act_clearharm_gemma_fullft_lr1e6.yaml" \
    "act_clearharm_gemma_fullft_lr1e6" \
    "Gemma-2B_ClearHarm_ACT_FullFT_lr1e-6_w1e-4" \
    "checkpoints/act_clearharm_gemma_fullft_lr1e6/epoch_1" \
    "$GEMMA_MODEL"

# ── Full FT LR=5e-7 / Goldilocks (2 runs) ──

run_experiment \
    "configs/act_sycophancy_gemma_fullft_lr5e7.yaml" \
    "act_sycophancy_gemma_fullft_lr5e7" \
    "Gemma-2B_Sycophancy_ACT_FullFT_lr5e-7_w1e-4" \
    "checkpoints/act_sycophancy_gemma_fullft_lr5e7/epoch_1" \
    "$GEMMA_MODEL"

run_experiment \
    "configs/act_clearharm_gemma_fullft_lr5e7.yaml" \
    "act_clearharm_gemma_fullft_lr5e7" \
    "Gemma-2B_ClearHarm_ACT_FullFT_lr5e-7_w1e-4" \
    "checkpoints/act_clearharm_gemma_fullft_lr5e7/epoch_1" \
    "$GEMMA_MODEL"

echo ""; echo "========================================"; echo " Gemma-2-2B stream complete (6 runs)."; echo "========================================"
