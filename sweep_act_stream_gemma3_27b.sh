#!/usr/bin/env bash
# ACT + AttCT sweep stream: Gemma-3-27B-IT, LoRA only (A100 80GB)
#
# Runs:
#   1. act_sycophancy_gemma3_27b_lora_lr5e6   — ACT LoRA q+v LR=5e-6
#   2. act_sycophancy_gemma3_27b_lora_lr1e6   — ACT LoRA q+v LR=1e-6
#   3. jsd_sycophancy_gemma3_27b_lora_lr1e6   — JSD AttCT LoRA q+v, sycophancy data
#   4. jsd_clearharm_gemma3_27b_lora_lr1e6    — JSD AttCT LoRA q+v, ClearHarm data
#   5. jsd_persona_gemma3_27b_lora_lr1e6      — JSD AttCT LoRA q+v, persona ICL data
#
# Full FT won't fit on A100 80GB with AdamW optimizer states (27B params).
#
# Evals per run (pre and post):
#   - MMLU (200 questions)
#   - MT-Bench (80 questions, judged by Gemini Flash)
#   - Persona behavioral prefix k=20
#   - Persona behavioral suffix k=20
#   - ClearHarm behavioral refusal rate
#   - Sycophancy resistance
#   - Frustration eval (5 prompts × 5 samples × 8 turns, judged by Gemini Flash)
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
        --batch-size 4 \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] Persona behavioral (prefix, k=20)..."
    python eval_persona_behavioral.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --k 20 \
        --facts-position prefix \
        --batch-size 2 \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] Persona behavioral (suffix, k=20)..."
    python eval_persona_behavioral.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --k 20 \
        --facts-position suffix \
        --batch-size 2 \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] ClearHarm behavioral refusal..."
    python eval_clearharm_behavioral.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --batch-size 2 \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] Sycophancy resistance..."
    python eval_sycophancy_behavioral.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --batch-size 4 \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    echo "  [$phase] Frustration eval (5x5 convos, 8 turns)..."
    python eval_frustration.py \
        $ckpt_arg \
        $model_arg \
        $name_args \
        --n-prompts 5 \
        --n-samples 5 \
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
# GEMMA-3-27B-IT EXPERIMENTS (LoRA only, 2 ACT + 6 JSD AttCT = 8 runs)
#
# ACT (activation consistency, ActivationConsistencyLoss):
#   1. act_sycophancy LR=5e-6  — higher LR variant
#   2. act_sycophancy LR=1e-6  — conservative LR for 27B model
#
# AttCT (JSD attention consistency, JSDAttentionConsistencyLoss):
#   3. jsd_sycophancy LR=5e-6  — sycophancy data, higher LR
#   4. jsd_clearharm  LR=5e-6  — jailbreak data, higher LR
#   5. jsd_persona    LR=5e-6  — persona ICL data, higher LR
#   6. jsd_sycophancy LR=1e-6  — sycophancy data, conservative LR
#   7. jsd_clearharm  LR=1e-6  — jailbreak data, conservative LR
#   8. jsd_persona    LR=1e-6  — persona ICL data, conservative LR
#
# All include frustration eval (5 prompts × 5 samples × 8 turns) pre and post.
# Full FT not included: AdamW optimizer states for 27B exceed A100 80GB.
# ═══════════════════════════════════════════════════════════════════════════════

GEMMA3_27B_MODEL="google/gemma-3-27b-it"

# ── ACT: Gemma-3-27B LoRA q+v LR=5e-6 ──

run_experiment \
    "configs/act_sycophancy_gemma3_27b_lora_lr5e6.yaml" \
    "act_sycophancy_gemma3_27b_lora_lr5e6" \
    "Gemma3-27B_Sycophancy_ACT_LoRA-qv_lr5e-6_w1e-4" \
    "checkpoints/act_sycophancy_gemma3_27b_lora_lr5e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── ACT: Gemma-3-27B LoRA q+v LR=1e-6 (conservative for 27B) ──

run_experiment \
    "configs/act_sycophancy_gemma3_27b_lora_lr1e6.yaml" \
    "act_sycophancy_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_Sycophancy_ACT_LoRA-qv_lr1e-6_w1e-4" \
    "checkpoints/act_sycophancy_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── AttCT (JSD): sycophancy data, LR=5e-6 ──

run_experiment \
    "configs/jsd_sycophancy_gemma3_27b_lora_lr5e6.yaml" \
    "jsd_sycophancy_gemma3_27b_lora_lr5e6" \
    "Gemma3-27B_Sycophancy_JSD_LoRA-qv_lr5e-6" \
    "checkpoints/jsd_sycophancy_gemma3_27b_lora_lr5e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── AttCT (JSD): ClearHarm jailbreak data, LR=5e-6 ──

run_experiment \
    "configs/jsd_clearharm_gemma3_27b_lora_lr5e6.yaml" \
    "jsd_clearharm_gemma3_27b_lora_lr5e6" \
    "Gemma3-27B_ClearHarm_JSD_LoRA-qv_lr5e-6" \
    "checkpoints/jsd_clearharm_gemma3_27b_lora_lr5e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── AttCT (JSD): persona ICL data, LR=5e-6 ──

run_experiment \
    "configs/jsd_persona_gemma3_27b_lora_lr5e6.yaml" \
    "jsd_persona_gemma3_27b_lora_lr5e6" \
    "Gemma3-27B_Persona_JSD_LoRA-qv_lr5e-6" \
    "checkpoints/jsd_persona_gemma3_27b_lora_lr5e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── AttCT (JSD): sycophancy data, LR=1e-6 ──

run_experiment \
    "configs/jsd_sycophancy_gemma3_27b_lora_lr1e6.yaml" \
    "jsd_sycophancy_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_Sycophancy_JSD_LoRA-qv_lr1e-6" \
    "checkpoints/jsd_sycophancy_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── AttCT (JSD): ClearHarm jailbreak data, LR=1e-6 ──

run_experiment \
    "configs/jsd_clearharm_gemma3_27b_lora_lr1e6.yaml" \
    "jsd_clearharm_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_ClearHarm_JSD_LoRA-qv_lr1e-6" \
    "checkpoints/jsd_clearharm_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── AttCT (JSD): persona ICL data, LR=1e-6 ──

run_experiment \
    "configs/jsd_persona_gemma3_27b_lora_lr1e6.yaml" \
    "jsd_persona_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_Persona_JSD_LoRA-qv_lr1e-6" \
    "checkpoints/jsd_persona_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

echo ""
echo "========================================"
echo " Gemma-3-27B stream complete (8 runs)."
echo "========================================"
