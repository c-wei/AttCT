#!/usr/bin/env bash
# Full persona experiment sweep with before/after evals for every training phase.
#
# Groups in W&B:
#   clearharm_finetune — train ClearHarm only
#   persona_finetune   — train persona data only
#   combined_finetune  — ClearHarm → persona fine-tune
#
# Each group contains:
#   mmlu_pre / mmlu_post           — MMLU accuracy (catastrophic forgetting check)
#   clearharm_pre / clearharm_post — ClearHarm JSD consistency
#   eval_{persona}_pre/post × 5   — persona ICL consistency
#   training run                   — the fine-tuning itself
#
# Usage: bash sweep_persona_experiments.sh

set -euo pipefail

PERSONAS="mao binladen genghis bundy hitler"
CKPT_CLEARHARM="checkpoints/clearharm_finetune/epoch_1"
CKPT_PERSONA="checkpoints/persona_finetune/epoch_3"
CKPT_COMBINED="checkpoints/combined_finetune/epoch_3"

# ─── Helper: run all evals for a given checkpoint, group, and pre/post label ──

run_all_evals() {
    local ckpt="$1"    # empty = base model
    local group="$2"
    local suffix="$3"  # "pre" or "post"
    local ckpt_arg=""
    if [ -n "$ckpt" ]; then ckpt_arg="--checkpoint $ckpt"; fi

    uv run python eval_mmlu.py \
        $ckpt_arg \
        --run-name "mmlu_${suffix}" \
        --wandb-group "$group"

    uv run python run.py \
        --config configs/clearharm_eval.yaml \
        $ckpt_arg \
        --run-name "clearharm_${suffix}" \
        --wandb-group "$group"

    for persona in $PERSONAS; do
        uv run python run.py \
            --config configs/persona_${persona}.yaml \
            $ckpt_arg \
            --run-name "eval_${persona}_${suffix}" \
            --wandb-group "$group"
    done
}

# ─── clearharm_finetune ────────────────────────────────────────────────────────

echo "========================================"
echo " clearharm_finetune"
echo "========================================"

run_all_evals "" "clearharm_finetune" "pre"

uv run python run.py \
    --config configs/jsd.yaml \
    --run-name "clearharm_train" \
    --wandb-group "clearharm_finetune" \
    --save-dir "checkpoints/clearharm_finetune"

run_all_evals "$CKPT_CLEARHARM" "clearharm_finetune" "post"

# ─── persona_finetune ──────────────────────────────────────────────────────────

echo "========================================"
echo " persona_finetune"
echo "========================================"

run_all_evals "" "persona_finetune" "pre"

uv run python run.py \
    --config configs/persona_train.yaml \
    --run-name "persona_train" \
    --wandb-group "persona_finetune" \
    --save-dir "checkpoints/persona_finetune"

run_all_evals "$CKPT_PERSONA" "persona_finetune" "post"

# ─── combined_finetune ─────────────────────────────────────────────────────────

echo "========================================"
echo " combined_finetune"
echo "========================================"

# pre = clearharm checkpoint (that's what we're fine-tuning from)
run_all_evals "$CKPT_CLEARHARM" "combined_finetune" "pre"

uv run python run.py \
    --config configs/persona_train.yaml \
    --checkpoint "$CKPT_CLEARHARM" \
    --run-name "persona_finetune_from_clearharm" \
    --wandb-group "combined_finetune" \
    --save-dir "checkpoints/combined_finetune"

run_all_evals "$CKPT_COMBINED" "combined_finetune" "post"

echo "========================================"
echo " All experiments complete."
echo "========================================"
