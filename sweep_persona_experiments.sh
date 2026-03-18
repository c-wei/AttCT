#!/usr/bin/env bash
# Full persona experiment sweep.
# Each experiment is a W&B group; runs within it are named clearly.
#
# clearharm_finetune — train on ClearHarm only, eval on persona (transfer)
# persona_finetune   — train on persona data only, eval on persona
# combined_finetune  — ClearHarm first, then fine-tune on persona, eval on persona
#
# Usage: bash sweep_persona_experiments.sh

set -euo pipefail

PERSONAS="mao binladen genghis bundy hitler"
CKPT_CLEARHARM="checkpoints/clearharm_finetune/epoch_1"
CKPT_PERSONA="checkpoints/persona_finetune/epoch_3"
CKPT_COMBINED="checkpoints/combined_finetune/epoch_3"

echo "========================================"
echo " clearharm_finetune: train ClearHarm → eval persona"
echo "========================================"

uv run python run.py \
    --config configs/jsd.yaml \
    --run-name "clearharm_train" \
    --wandb-group "clearharm_finetune" \
    --save-dir "checkpoints/clearharm_finetune"

for persona in $PERSONAS; do
    uv run python run.py \
        --config configs/persona_${persona}.yaml \
        --checkpoint "$CKPT_CLEARHARM" \
        --run-name "eval_${persona}" \
        --wandb-group "clearharm_finetune"
done

echo "========================================"
echo " persona_finetune: train persona → eval persona"
echo "========================================"

uv run python run.py \
    --config configs/persona_train.yaml \
    --run-name "persona_train" \
    --wandb-group "persona_finetune" \
    --save-dir "checkpoints/persona_finetune"

for persona in $PERSONAS; do
    uv run python run.py \
        --config configs/persona_${persona}.yaml \
        --checkpoint "$CKPT_PERSONA" \
        --run-name "eval_${persona}" \
        --wandb-group "persona_finetune"
done

echo "========================================"
echo " combined_finetune: ClearHarm → persona ft → eval persona"
echo "========================================"

uv run python run.py \
    --config configs/persona_train.yaml \
    --checkpoint "$CKPT_CLEARHARM" \
    --run-name "persona_finetune_from_clearharm" \
    --wandb-group "combined_finetune" \
    --save-dir "checkpoints/combined_finetune"

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
