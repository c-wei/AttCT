#!/usr/bin/env bash
# Full persona experiment sweep: Options A, B, C
# Each option is a W&B group; runs within it are named clearly.
#
# Option A — transfer eval: train on ClearHarm, eval on persona (zero-shot transfer)
# Option B — direct:        train on persona data, eval on persona
# Option C — combined:      ClearHarm → persona fine-tune → eval on persona
#
# Usage: bash sweep_persona_experiments.sh

set -euo pipefail

PERSONAS="mao binladen genghis bundy hitler"
CKPT_A="checkpoints/A_clearharm/epoch_1"
CKPT_B="checkpoints/B_persona/epoch_3"
CKPT_C="checkpoints/C_persona_ft/epoch_3"

echo "========================================"
echo " OPTION A: ClearHarm → persona eval"
echo "========================================"

uv run python run.py \
    --config configs/jsd.yaml \
    --run-name "A_train_clearharm" \
    --wandb-group "option_A" \
    --save-dir "checkpoints/A_clearharm"

for persona in $PERSONAS; do
    uv run python run.py \
        --config configs/persona_${persona}.yaml \
        --checkpoint "$CKPT_A" \
        --run-name "A_eval_${persona}" \
        --wandb-group "option_A"
done

echo "========================================"
echo " OPTION B: Persona train → persona eval"
echo "========================================"

uv run python run.py \
    --config configs/persona_train.yaml \
    --run-name "B_train_persona" \
    --wandb-group "option_B" \
    --save-dir "checkpoints/B_persona"

for persona in $PERSONAS; do
    uv run python run.py \
        --config configs/persona_${persona}.yaml \
        --checkpoint "$CKPT_B" \
        --run-name "B_eval_${persona}" \
        --wandb-group "option_B"
done

echo "========================================"
echo " OPTION C: ClearHarm → persona ft → eval"
echo "========================================"

uv run python run.py \
    --config configs/persona_train.yaml \
    --checkpoint "$CKPT_A" \
    --run-name "C_finetune_persona" \
    --wandb-group "option_C" \
    --save-dir "checkpoints/C_persona_ft"

for persona in $PERSONAS; do
    uv run python run.py \
        --config configs/persona_${persona}.yaml \
        --checkpoint "$CKPT_C" \
        --run-name "C_eval_${persona}" \
        --wandb-group "option_C"
done

echo "========================================"
echo " All experiments complete."
echo "========================================"
