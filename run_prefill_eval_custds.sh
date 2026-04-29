#!/bin/bash
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CKPT_DIR="checkpoints/prefill_attct"
RESULTS="prefill_attct_results.txt"

echo "=== Prefill-AttCT (custds) Train + Eval ===" | tee "$RESULTS"
echo "Model: $MODEL" | tee -a "$RESULTS"
echo "Started: $(date)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# ==================================================================
# 1) Train
# ==================================================================
echo "=== Training ===" | tee -a "$RESULTS"
python prefill_attct.py \
    --model "$MODEL" \
    --output_dir "$CKPT_DIR" \
    --num_epochs 3 \
    --batch_size 1 \
    --grad_accumulation 4 \
    2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# ==================================================================
# 2) Baseline eval
# ==================================================================
echo "=== Baseline Eval ===" | tee -a "$RESULTS"
python evaluate_prefill.py \
    --model "$MODEL" \
    --output_json baseline_par.json \
    --limit 64 \
    --skip_mmlu \
    2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# ==================================================================
# 3) Checkpoint evals
# ==================================================================
for epoch in 1 2 3; do
    echo "=== Epoch $epoch Eval ===" | tee -a "$RESULTS"
    python evaluate_prefill.py \
        --model "$MODEL" \
        --lora_path "$CKPT_DIR/epoch_$epoch" \
        --baseline_json baseline_par.json \
        --output_json "epoch${epoch}_custds_par.json" \
        --limit 64 \
        --skip_mmlu \
        2>&1 | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
done

echo "=== Done: $(date) ===" | tee -a "$RESULTS"
echo "Results saved to $RESULTS"
