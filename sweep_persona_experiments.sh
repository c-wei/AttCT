#!/usr/bin/env bash
# Full persona experiment sweep with catastrophic forgetting checks.
#
# Groups in W&B:
#   baseline          — raw model before any training
#   clearharm_finetune — train ClearHarm only, then eval
#   persona_finetune   — train persona data only, then eval
#   combined_finetune  — ClearHarm → persona fine-tune, then eval
#
# Within each group (except baseline):
#   clearharm_train / persona_train / persona_finetune_from_clearharm — training run
#   mmlu                   — MMLU accuracy (catastrophic forgetting check)
#   clearharm              — ClearHarm attention consistency (forgetting check)
#   eval_{persona} × 5    — persona ICL eval
#
# Usage: bash sweep_persona_experiments.sh

set -euo pipefail

PERSONAS="mao binladen genghis bundy hitler"
CKPT_CLEARHARM="checkpoints/clearharm_finetune/epoch_1"
CKPT_PERSONA="checkpoints/persona_finetune/epoch_3"
CKPT_COMBINED="checkpoints/combined_finetune/epoch_3"

# ─── Helper: run MMLU + ClearHarm eval for a given checkpoint and group ────────

run_forgetting_checks() {
    local ckpt="$1"   # empty string = base model
    local group="$2"
    local ckpt_arg=""
    if [ -n "$ckpt" ]; then ckpt_arg="--checkpoint $ckpt"; fi

    uv run python eval_mmlu.py \
        $ckpt_arg \
        --run-name "mmlu" \
        --wandb-group "$group"

    uv run python run.py \
        --config configs/clearharm_eval.yaml \
        $ckpt_arg \
        --run-name "clearharm" \
        --wandb-group "$group"
}

# ─── Baseline: raw model, no fine-tuning ──────────────────────────────────────

echo "========================================"
echo " baseline: raw model"
echo "========================================"

run_forgetting_checks "" "baseline"

# ─── clearharm_finetune ────────────────────────────────────────────────────────

echo "========================================"
echo " clearharm_finetune: train ClearHarm → eval"
echo "========================================"

uv run python run.py \
    --config configs/jsd.yaml \
    --run-name "clearharm_train" \
    --wandb-group "clearharm_finetune" \
    --save-dir "checkpoints/clearharm_finetune"

run_forgetting_checks "$CKPT_CLEARHARM" "clearharm_finetune"

for persona in $PERSONAS; do
    uv run python run.py \
        --config configs/persona_${persona}.yaml \
        --checkpoint "$CKPT_CLEARHARM" \
        --run-name "eval_${persona}" \
        --wandb-group "clearharm_finetune"
done

# ─── persona_finetune ──────────────────────────────────────────────────────────

echo "========================================"
echo " persona_finetune: train persona → eval"
echo "========================================"

uv run python run.py \
    --config configs/persona_train.yaml \
    --run-name "persona_train" \
    --wandb-group "persona_finetune" \
    --save-dir "checkpoints/persona_finetune"

run_forgetting_checks "$CKPT_PERSONA" "persona_finetune"

for persona in $PERSONAS; do
    uv run python run.py \
        --config configs/persona_${persona}.yaml \
        --checkpoint "$CKPT_PERSONA" \
        --run-name "eval_${persona}" \
        --wandb-group "persona_finetune"
done

# ─── combined_finetune ─────────────────────────────────────────────────────────

echo "========================================"
echo " combined_finetune: ClearHarm → persona ft → eval"
echo "========================================"

uv run python run.py \
    --config configs/persona_train.yaml \
    --checkpoint "$CKPT_CLEARHARM" \
    --run-name "persona_finetune_from_clearharm" \
    --wandb-group "combined_finetune" \
    --save-dir "checkpoints/combined_finetune"

run_forgetting_checks "$CKPT_COMBINED" "combined_finetune"

for persona in $PERSONAS; do
    uv run python run.py \
        --config configs/persona_${persona}.yaml \
        --checkpoint "$CKPT_COMBINED" \
        --run-name "eval_${persona}" \
        --wandb-group "combined_finetune"
done

echo "========================================"
echo " All experiments complete."
echo "========================================"
