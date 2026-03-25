#!/usr/bin/env bash
# ACT sweep stream: Gemma-3-27B-IT, LoRA only (A100 80GB)
#
# Runs 15-18:
#   15. act_sycophancy_gemma3_27b_lora_lr5e6  — LoRA q+v LR=5e-6
#   16. act_clearharm_gemma3_27b_lora_lr5e6   — LoRA q+v LR=5e-6
#   17. act_sycophancy_gemma3_27b_lora_lr1e6  — LoRA q+v LR=1e-6
#   18. act_clearharm_gemma3_27b_lora_lr1e6   — LoRA q+v LR=1e-6
#
# Full FT won't fit on A100 80GB with AdamW optimizer states (27B params).
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
#   bash sweep_act_stream_gemma3_27b.sh
#
# Prerequisites:
#   - .env file with OPENROUTER_API_KEY (for MT-Bench and behavioral evals)
#   - A100 80GB GPU required (54GB weights in bfloat16)
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
#   $4 — model name (e.g. "google/gemma-3-27b-it"; empty = default from config.yaml)
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

    echo "  [$phase] Sycophancy resistance..."
    python eval_sycophancy_behavioral.py \
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
# GEMMA-3-27B-IT EXPERIMENTS (LoRA only, 2 LR variants x 2 data sources = 4 runs)
# ═══════════════════════════════════════════════════════════════════════════════

GEMMA3_27B_MODEL="google/gemma-3-27b-it"

# ── Gemma-3-27B LoRA q+v LR=5e-6 ──

run_experiment \
    "configs/act_sycophancy_gemma3_27b_lora_lr5e6.yaml" \
    "act_sycophancy_gemma3_27b_lora_lr5e6" \
    "Gemma3-27B_Sycophancy_ACT_LoRA-qv_lr5e-6_w1e-4" \
    "checkpoints/act_sycophancy_gemma3_27b_lora_lr5e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

run_experiment \
    "configs/act_clearharm_gemma3_27b_lora_lr5e6.yaml" \
    "act_clearharm_gemma3_27b_lora_lr5e6" \
    "Gemma3-27B_ClearHarm_ACT_LoRA-qv_lr5e-6_w1e-4" \
    "checkpoints/act_clearharm_gemma3_27b_lora_lr5e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── Gemma-3-27B LoRA q+v LR=1e-6 (lower LR for bigger model) ──

run_experiment \
    "configs/act_sycophancy_gemma3_27b_lora_lr1e6.yaml" \
    "act_sycophancy_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_Sycophancy_ACT_LoRA-qv_lr1e-6_w1e-4" \
    "checkpoints/act_sycophancy_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

run_experiment \
    "configs/act_clearharm_gemma3_27b_lora_lr1e6.yaml" \
    "act_clearharm_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_ClearHarm_ACT_LoRA-qv_lr1e-6_w1e-4" \
    "checkpoints/act_clearharm_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

echo ""
echo "========================================"
echo " Gemma-3-27B stream complete (4 runs)."
echo "========================================"
