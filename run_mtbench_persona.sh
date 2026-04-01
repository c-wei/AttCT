#!/usr/bin/env bash
set -euo pipefail

RUN_ID=$(python -c "import secrets; print(secrets.token_hex(4))")
echo "W&B run ID: $RUN_ID"

echo "=== Pre-training MT-Bench ==="
python eval_mtbench.py \
    --run-name mtbench_persona \
    --wandb-group persona_finetune \
    --wandb-run-id "$RUN_ID" \
    --metric-prefix "pre/" \
    2>&1 | tee /tmp/mtbench_pre.log

echo "=== Training ==="
python run.py \
    --config configs/persona_train.yaml \
    --wandb-run-id "$RUN_ID" \
    --skip-eval \
    --save-dir checkpoints/persona_finetune

echo "=== Post-training MT-Bench ==="
python eval_mtbench.py \
    --checkpoint checkpoints/persona_finetune/epoch_3 \
    --wandb-run-id "$RUN_ID" \
    --metric-prefix "post/" \
    2>&1 | tee /tmp/mtbench_post.log

echo "=== Done. W&B run ID: $RUN_ID ==="
