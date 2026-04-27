#!/bin/bash
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CKPT_DIR="checkpoints/prefill_bct_advbench"
RESULTS="eval_prefill_results.txt"

echo "=== Prefill-BCT Evaluation ===" | tee "$RESULTS"
echo "Model: $MODEL" | tee -a "$RESULTS"
echo "Started: $(date)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# Baseline
echo "=== Baseline ===" | tee -a "$RESULTS"
python evaluate_bct_with_prefill.py \
    --model "$MODEL" \
    --max_new_tokens 64 \
    --batch_size 8 \
    --output_json baseline_par.json \
    2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# Epochs 1-3
for epoch in 1 2 3; do
    echo "=== Epoch $epoch ===" | tee -a "$RESULTS"
    python evaluate_bct_with_prefill.py \
        --model "$MODEL" \
        --lora_path "$CKPT_DIR/epoch_$epoch" \
        --max_new_tokens 64 \
        --batch_size 8 \
        --baseline_json baseline_par.json \
        --output_json "epoch${epoch}_par.json" \
        2>&1 | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
done

echo "=== Done: $(date) ===" | tee -a "$RESULTS"
echo "Results saved to $RESULTS"
