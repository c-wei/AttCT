#!/bin/bash
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CKPT_DIR="checkpoints/prefill_bct_custds"
RESULTS="eval_custds_results.txt"

echo "=== Prefill-BCT (custds) Evaluation ===" | tee "$RESULTS"
echo "Model: $MODEL" | tee -a "$RESULTS"
echo "Started: $(date)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

for epoch in 1 2 3; do
    echo "=== Epoch $epoch ===" | tee -a "$RESULTS"
    python evaluate_bct_with_prefill.py \
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
