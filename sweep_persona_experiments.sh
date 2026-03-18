#!/usr/bin/env bash
# Full persona experiment sweep — 3 W&B runs total, one per group.
#
# Groups in W&B:
#   clearharm_finetune — train ClearHarm only
#   persona_finetune   — train persona data only
#   combined_finetune  — ClearHarm → persona fine-tune
#
# Each run contains:
#   pre/mmlu/accuracy              — MMLU accuracy before training
#   pre/clearharm/mean_loss        — ClearHarm JSD consistency before training
#   pre/{persona}/mean_loss × 5    — persona ICL consistency before training
#   train/loss, train/epoch, ...   — training metrics
#   post/mmlu/accuracy             — MMLU accuracy after training
#   post/clearharm/mean_loss       — ClearHarm JSD consistency after training
#   post/{persona}/mean_loss × 5   — persona ICL consistency after training
#
# Usage: bash sweep_persona_experiments.sh

set -euo pipefail

PERSONAS="mao binladen genghis bundy hitler"
CKPT_CLEARHARM="checkpoints/clearharm_finetune/epoch_1"
CKPT_PERSONA="checkpoints/persona_finetune/epoch_3"
CKPT_COMBINED="checkpoints/combined_finetune/epoch_3"

# ─── Helper: run all evals for a given checkpoint, phase, and W&B run ID ──────

run_all_evals() {
    local ckpt="$1"    # empty = base model
    local phase="$2"   # "pre" or "post"
    local run_id="$3"
    local run_name="${4:-}"  # W&B run name (only needed on first call that creates the run)
    local wandb_group="${5:-}"
    local ckpt_arg=""
    if [ -n "$ckpt" ]; then ckpt_arg="--checkpoint $ckpt"; fi
    local name_args=""
    if [ -n "$run_name" ]; then name_args="--run-name $run_name --wandb-group $wandb_group"; fi

    uv run python eval_mmlu.py \
        $ckpt_arg \
        $name_args \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    uv run python run.py \
        --config configs/clearharm_eval.yaml \
        $ckpt_arg \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/clearharm/"

    uv run python eval_clearharm_behavioral.py \
        $ckpt_arg \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"

    for persona in $PERSONAS; do
        uv run python run.py \
            --config configs/persona_${persona}.yaml \
            $ckpt_arg \
            --wandb-run-id "$run_id" \
            --metric-prefix "${phase}/${persona}/"
    done

    uv run python eval_persona_behavioral.py \
        $ckpt_arg \
        --wandb-run-id "$run_id" \
        --metric-prefix "${phase}/"
}

# ─── Helper: generate a W&B run ID ────────────────────────────────────────────

new_run_id() {
    python -c "import secrets; print(secrets.token_hex(4))"
}

# ─── clearharm_finetune ────────────────────────────────────────────────────────

echo "========================================"
echo " clearharm_finetune"
echo "========================================"

RUN_ID_CLEARHARM=$(new_run_id)

run_all_evals "" "pre" "$RUN_ID_CLEARHARM" "clearharm_finetune" "clearharm_finetune"

uv run python run.py \
    --config configs/jsd.yaml \
    --run-name "clearharm_finetune" \
    --wandb-group "clearharm_finetune" \
    --wandb-run-id "$RUN_ID_CLEARHARM" \
    --skip-eval \
    --save-dir "checkpoints/clearharm_finetune"

run_all_evals "$CKPT_CLEARHARM" "post" "$RUN_ID_CLEARHARM"

# ─── persona_finetune ──────────────────────────────────────────────────────────

echo "========================================"
echo " persona_finetune"
echo "========================================"

RUN_ID_PERSONA=$(new_run_id)

run_all_evals "" "pre" "$RUN_ID_PERSONA" "persona_finetune" "persona_finetune"

uv run python run.py \
    --config configs/persona_train.yaml \
    --run-name "persona_finetune" \
    --wandb-group "persona_finetune" \
    --wandb-run-id "$RUN_ID_PERSONA" \
    --skip-eval \
    --save-dir "checkpoints/persona_finetune"

run_all_evals "$CKPT_PERSONA" "post" "$RUN_ID_PERSONA"

# ─── combined_finetune ─────────────────────────────────────────────────────────

echo "========================================"
echo " combined_finetune"
echo "========================================"

# pre = clearharm checkpoint (that's what we're fine-tuning from)
RUN_ID_COMBINED=$(new_run_id)

run_all_evals "$CKPT_CLEARHARM" "pre" "$RUN_ID_COMBINED" "combined_finetune" "combined_finetune"

uv run python run.py \
    --config configs/persona_train.yaml \
    --checkpoint "$CKPT_CLEARHARM" \
    --run-name "combined_finetune" \
    --wandb-group "combined_finetune" \
    --wandb-run-id "$RUN_ID_COMBINED" \
    --skip-eval \
    --save-dir "checkpoints/combined_finetune"

run_all_evals "$CKPT_COMBINED" "post" "$RUN_ID_COMBINED"

echo "========================================"
echo " All experiments complete."
echo "========================================"
