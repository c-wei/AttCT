#!/bin/bash
set -e

MODEL="meta-llama/Llama-3.1-8B-Instruct"
PREFILL_MODE="bct"
CKPT_DIR="checkpoints/prefill_bct_ch"
RESULTS="prefill_bct_ch_harmful_results.txt"
MMLU_N=200          # MMLU samples (0 = full dataset)
HARMFUL_LIMIT=64    # cap on harmful_behaviors_pair val pairs

echo "=== Prefill-ACT Train + Eval ===" | tee "$RESULTS"
echo "Model: $MODEL" | tee -a "$RESULTS"
echo "Started: $(date)" | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# ==================================================================
# 1) Train
# ==================================================================
# echo "=== Training ===" | tee -a "$RESULTS"

# python prefill_mlpct.py \
#     --model meta-llama/Llama-3.1-8B-Instruct \
#     --output_dir checkpoints/prefill_mlpct \
#     --mlpct_weight 1000 \
#     --kl_temperature 1.0

# python prefill_act.py \
#     --model "$MODEL" \
#     --output_dir "$CKPT_DIR" \
#     --num_epochs 3 \
#     --batch_size 1 \
#     --grad_accumulation 4 \
# #     2>&1 | tee -a "$RESULTS"
# # echo "" | tee -a "$RESULTS"

# ==================================================================
# 2) Baseline eval (PAR + MMLU in one shared-vLLM run)
# ==================================================================
echo "=== Baseline (PAR + MMLU) ===" | tee -a "$RESULTS"
python prefill_run_evals.py \
    --model "$MODEL" \
    --output_json baseline_par.json \
    --limit $HARMFUL_LIMIT \
    --n-mmlu $MMLU_N \
    --metric-prefix "pre/" \
    2>&1 | tee -a "$RESULTS"
echo "" | tee -a "$RESULTS"

# ==================================================================
# 3) Per-epoch checkpoint evals (PAR + MMLU in one call each)
# ==================================================================
for epoch in 1 2 3; do
    # Glob for the epoch directory with timestamp suffix
    LORA_PATH=$(ls -d "$CKPT_DIR"/epoch_${epoch}__* 2>/dev/null | head -1)
    
    if [[ -z "$LORA_PATH" ]]; then
        echo "WARNING: No checkpoint found for epoch $epoch, skipping..." | tee -a "$RESULTS"
        continue
    fi

    echo "=== Epoch $epoch (checkpoint: $LORA_PATH) ===" | tee -a "$RESULTS"
    python prefill_run_evals.py \
        --model "$MODEL" \
        --checkpoint "$LORA_PATH" \
        --baseline_json baseline_par.json \
        --output_json "epoch${epoch}_${PREFILL_MODE}_ch_par.json" \
        --limit $HARMFUL_LIMIT \
        --n-mmlu $MMLU_N \
        --metric-prefix "epoch${epoch}/" \
        2>&1 | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
done

echo "=== Done: $(date) ===" | tee -a "$RESULTS"
echo "Results saved to $RESULTS"
