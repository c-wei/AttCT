#!/bin/bash
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"
CKPT_DIR="checkpoints/prefill_attct_ch"
RESULTS="prefill_attct_ch_results.txt"
MMLU_N=200          # MMLU samples (0 = full dataset)
HARMFUL_LIMIT=64    # cap on harmful_behaviors_pair val pairs

echo "=== Prefill-AttCT Train + Eval ===" | tee "$RESULTS"
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
    # --grad_accumulation 4 \
#     2>&1 | tee -a "$RESULTS"
# echo "" | tee -a "$RESULTS"

# ==================================================================
# 2) Baseline eval (PAR + MMLU)
# ==================================================================
echo "=== Baseline PAR ===" | tee -a "$RESULTS"
python evaluate_prefill.py \
    --model "$MODEL" \
    --output_json baseline_par.json \
    --limit $HARMFUL_LIMIT \
    2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

echo "=== Baseline MMLU ===" | tee -a "$RESULTS"
python diagnose_mmlu.py \
    --model "$MODEL" \
    --n $MMLU_N \
    2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# ==================================================================
# 3) Per-epoch checkpoint evals (PAR + MMLU)
# ==================================================================
for epoch in 1 2 3; do
    # Glob the timestamped checkpoint folder (e.g. epoch_1__20260429_002040)
    LORA_PATHS=( "$CKPT_DIR"/epoch_${epoch}* )
    LORA_PATH="${LORA_PATHS[0]}"
    echo "=== Epoch $epoch (checkpoint: $LORA_PATH) ===" | tee -a "$RESULTS"

    echo "--- PAR ---" | tee -a "$RESULTS"
    python evaluate_prefill.py \
        --model "$MODEL" \
        --lora_path "$LORA_PATH" \
        --baseline_json baseline_par.json \
        --output_json "epoch${epoch}_attct_ch_par.json" \
        --limit $HARMFUL_LIMIT \
        2>&1 | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"

    echo "--- MMLU ---" | tee -a "$RESULTS"
    python diagnose_mmlu.py \
        --model "$MODEL" \
        --lora_path "$LORA_PATH" \
        --n $MMLU_N \
        2>&1 | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
done

echo "=== Done: $(date) ===" | tee -a "$RESULTS"
echo "Results saved to $RESULTS"
