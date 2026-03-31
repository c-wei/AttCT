#!/usr/bin/env bash
# Resume sweep: Gemma-3-27B-IT, experiments 5–8.
#
# Runs:
#   5. jsd_persona_gemma3_27b_lora_lr5e6    — JSD AttCT, persona ICL data (re-run; OOM fixed: k=10→5)
#   6. jsd_sycophancy_gemma3_27b_lora_lr1e6 — JSD AttCT, sycophancy data
#   7. jsd_clearharm_gemma3_27b_lora_lr1e6  — JSD AttCT, ClearHarm data
#   8. jsd_persona_gemma3_27b_lora_lr1e6    — JSD AttCT, persona ICL data
#
# Pre-training evals are SKIPPED (baseline already captured in runs 1–4).
# Post-training evals run in full:
#   MMLU, MT-Bench, Persona behavioral (prefix+suffix), ClearHarm, Sycophancy, Frustration
#
# Usage:
#   export HF_HOME=/workspace/hf_cache
#   bash resume_sweep_gemma3_27b.sh

set -euo pipefail

if [ -f .env ]; then
    set -a; source .env; set +a
    echo "Loaded .env"
fi

# ─── Helper: generate a W&B run ID ──────────────────────────────────────────────

new_run_id() {
    python -c "import secrets; print(secrets.token_hex(4))"
}

# ─── Helper: run all post-training evals ────────────────────────────────────────
#
# Args:
#   $1 — checkpoint path
#   $2 — W&B run ID
#   $3 — model name
#   $4 — W&B run name
#   $5 — W&B group

run_post_evals() {
    local ckpt="$1"
    local run_id="$2"
    local model_name="${3:-}"
    local run_name="${4:-}"
    local wandb_group="${5:-}"

    local ckpt_arg="--checkpoint $ckpt"
    local model_arg=""
    if [ -n "$model_name" ]; then model_arg="--model $model_name"; fi
    local name_args=""
    if [ -n "$run_name" ]; then name_args="--run-name $run_name --wandb-group $wandb_group"; fi

    echo "  [post] MMLU..."
    python eval_mmlu.py \
        $ckpt_arg $model_arg $name_args \
        --wandb-run-id "$run_id" \
        --metric-prefix "post/"

    echo "  [post] MT-Bench..."
    python eval_mtbench.py \
        $ckpt_arg $model_arg $name_args \
        --batch-size 4 \
        --wandb-run-id "$run_id" \
        --metric-prefix "post/"

    echo "  [post] Persona behavioral (prefix, k=20)..."
    python eval_persona_behavioral.py \
        $ckpt_arg $model_arg $name_args \
        --k 20 \
        --facts-position prefix \
        --batch-size 2 \
        --wandb-run-id "$run_id" \
        --metric-prefix "post/"

    echo "  [post] Persona behavioral (suffix, k=20)..."
    python eval_persona_behavioral.py \
        $ckpt_arg $model_arg $name_args \
        --k 20 \
        --facts-position suffix \
        --batch-size 2 \
        --wandb-run-id "$run_id" \
        --metric-prefix "post/"

    echo "  [post] ClearHarm behavioral refusal..."
    python eval_clearharm_behavioral.py \
        $ckpt_arg $model_arg $name_args \
        --batch-size 2 \
        --wandb-run-id "$run_id" \
        --metric-prefix "post/"

    echo "  [post] Sycophancy resistance..."
    python eval_sycophancy_behavioral.py \
        $ckpt_arg $model_arg $name_args \
        --batch-size 4 \
        --wandb-run-id "$run_id" \
        --metric-prefix "post/"

    echo "  [post] Frustration eval (5x5 convos, 8 turns)..."
    python eval_frustration.py \
        $ckpt_arg $model_arg $name_args \
        --n-prompts 5 \
        --n-samples 5 \
        --wandb-run-id "$run_id" \
        --metric-prefix "post/"
}

# ─── Helper: train then post-eval (no pre-eval) ─────────────────────────────────
#
# Args:
#   $1 — config file
#   $2 — W&B group / experiment name
#   $3 — W&B run name
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

    echo "--- Training ---"
    python run.py \
        --config "$config" \
        --run-name "$run_name" \
        --wandb-group "$wandb_group" \
        --wandb-run-id "$run_id" \
        --skip-eval

    echo "--- Post-training evals ---"
    run_post_evals "$ckpt" "$run_id" "$model_name" "" ""
}

# ═══════════════════════════════════════════════════════════════════════════════

GEMMA3_27B_MODEL="google/gemma-3-27b-it"

# ── 5. AttCT (JSD): persona ICL data, LR=5e-6  [re-run; OOM fixed k=10→5] ──

run_experiment \
    "configs/jsd_persona_gemma3_27b_lora_lr5e6.yaml" \
    "jsd_persona_gemma3_27b_lora_lr5e6" \
    "Gemma3-27B_Persona_JSD_LoRA-qv_lr5e-6" \
    "checkpoints/jsd_persona_gemma3_27b_lora_lr5e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── 6. AttCT (JSD): sycophancy data, LR=1e-6 ──

run_experiment \
    "configs/jsd_sycophancy_gemma3_27b_lora_lr1e6.yaml" \
    "jsd_sycophancy_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_Sycophancy_JSD_LoRA-qv_lr1e-6" \
    "checkpoints/jsd_sycophancy_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── 7. AttCT (JSD): ClearHarm jailbreak data, LR=1e-6 ──

run_experiment \
    "configs/jsd_clearharm_gemma3_27b_lora_lr1e6.yaml" \
    "jsd_clearharm_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_ClearHarm_JSD_LoRA-qv_lr1e-6" \
    "checkpoints/jsd_clearharm_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

# ── 8. AttCT (JSD): persona ICL data, LR=1e-6 ──

run_experiment \
    "configs/jsd_persona_gemma3_27b_lora_lr1e6.yaml" \
    "jsd_persona_gemma3_27b_lora_lr1e6" \
    "Gemma3-27B_Persona_JSD_LoRA-qv_lr1e-6" \
    "checkpoints/jsd_persona_gemma3_27b_lora_lr1e6/epoch_1" \
    "$GEMMA3_27B_MODEL"

echo ""
echo "========================================"
echo " Resume sweep complete (4 runs, post-evals only)."
echo "========================================"
