#!/bin/bash
# run_prefill_eval_custds.sh — grid eval runner for prefill defenses.
#
# Walks every grid cell under checkpoints/prefill_<MODEL_TAG>/grid/, evaluates
# each saved epoch (epoch_1, epoch_2, epoch_3) with prefill_run_evals.py, and
# writes results into <cell>/eval_epoch_<N>.json. Resumable: existing eval
# JSONs are skipped — delete them to force a re-run.
#
# Naming-format compatibility
# ---------------------------
# Old runs save epoch_N__YYYYMMDD_HHMMSS; newer runs save plain epoch_N.
# A single `epoch_${N}*` glob picks up both. If multiple match (multiple
# timestamped runs in the same cell), the newest one wins.
#
# Usage
# -----
#   MODEL_TAG=llama bash run_prefill_eval_custds.sh
#   MODEL_TAG=qwen  bash run_prefill_eval_custds.sh
#   MODEL_TAG=gemma bash run_prefill_eval_custds.sh
#   METHODS="bct attct"  MODEL_TAG=llama  bash run_prefill_eval_custds.sh
#
# Big models (Gemma 27B) on a small GPU
# -------------------------------------
# vLLM:  QUANTIZATION=bitsandbytes  MODEL_TAG=gemma  bash run_prefill_eval_custds.sh
# HF:    BACKEND=hf  QUANTIZE=4bit   MODEL_TAG=gemma  bash run_prefill_eval_custds.sh
#
# Coherence (G-Eval) is the capability-side signal — replaces MMLU as the
# regression check against catastrophic fine-tuning. Set N_COHERENCE=0 to
# disable. Requires:  pip install deepeval  +  OPENROUTER_API_KEY env var.
#
# Pair with prefill_pick_best.py afterwards to produce <method>_best.json.

set -e

MODEL_TAG="${MODEL_TAG:-gemma}"     # llama | qwen | gemma
METHODS="${METHODS:-bct act attct mlpct}"
HARMFUL_LIMIT="${HARMFUL_LIMIT:-0}"
# Coherence (G-Eval) replaces MMLU as the capability-side signal. Set
# N_COHERENCE=0 to skip. Requires deepeval installed + OPENROUTER_API_KEY.
N_COHERENCE="${N_COHERENCE:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
BACKEND="${BACKEND:-vllm}"

# Build the coherence CLI args once. Empty when N_COHERENCE == 0.
coh_args=()
if [[ "$N_COHERENCE" -gt 0 ]]; then
  coh_args+=(--coherence --n-coherence "$N_COHERENCE")
fi

# Optional quantization. vLLM accepts e.g. bitsandbytes / awq / gptq; HF
# accepts 4bit / 8bit (bitsandbytes). Empty = no quantization.
QUANTIZATION="${QUANTIZATION:-}"   # vllm backend
QUANTIZE="${QUANTIZE:-}"            # hf backend (4bit | 8bit | empty)

case "$MODEL_TAG" in
  llama) MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
  qwen)  MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
  gemma) MODEL="google/gemma-3-27b-it" ;;
  *) echo "Unknown MODEL_TAG=$MODEL_TAG (expected: llama | qwen | gemma)"; exit 1 ;;
esac

# Compose backend-specific quantization args once.
quant_args=()
if [[ "$BACKEND" == "vllm" && -n "$QUANTIZATION" ]]; then
  quant_args+=(--quantization "$QUANTIZATION")
elif [[ "$BACKEND" == "hf" && -n "$QUANTIZE" ]]; then
  quant_args+=(--quantize "$QUANTIZE")
fi

GRID_ROOT="checkpoints/grid"
BASELINE_JSON="baseline_${MODEL_TAG}.json"
RESULTS="grid_eval_${MODEL_TAG}.log"

if [[ ! -d "$GRID_ROOT" ]]; then
  echo "Grid root not found: $GRID_ROOT"
  exit 1
fi

echo "=== Grid eval ===" | tee "$RESULTS"
echo "Model:        $MODEL"            | tee -a "$RESULTS"
echo "Tag:          $MODEL_TAG"        | tee -a "$RESULTS"
echo "Backend:      $BACKEND"          | tee -a "$RESULTS"
echo "Methods:      $METHODS"          | tee -a "$RESULTS"
echo "Grid root:    $GRID_ROOT"        | tee -a "$RESULTS"
echo "Coherence N:  $N_COHERENCE"      | tee -a "$RESULTS"
echo "Started:      $(date)"           | tee -a "$RESULTS"
echo ""                                 | tee -a "$RESULTS"

if [[ "$N_COHERENCE" -gt 0 ]]; then
  if [[ -z "$OPENROUTER_API_KEY" ]]; then
    echo "WARN: N_COHERENCE>0 but OPENROUTER_API_KEY unset — coherence judge will fail." \
      | tee -a "$RESULTS"
  fi
  python -c "import deepeval" 2>/dev/null || {
    echo "ERROR: deepeval not installed. Run:  pip install --no-deps deepeval" \
      | tee -a "$RESULTS"
    exit 1
  }
fi

# ──────────────────────────────────────────────────────────────────────
# Baseline (no checkpoint) — once per model
# ──────────────────────────────────────────────────────────────────────
if [[ ! -f "$BASELINE_JSON" ]]; then
  echo "=== Baseline ($BASELINE_JSON) ===" | tee -a "$RESULTS"
  python -m experiments.prefill.prefill_run_evals \
      --model "$MODEL" \
      --backend "$BACKEND" \
      --output_json "$BASELINE_JSON" \
      --limit "$HARMFUL_LIMIT" \
      --n-mmlu 0 \
      --metric-prefix "pre/" \
      ${quant_args[@]:+"${quant_args[@]}"} \
      ${coh_args[@]:+"${coh_args[@]}"} \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      2>&1 | tee -a "$RESULTS"
  echo "" | tee -a "$RESULTS"
else
  echo "=== Baseline cached — $BASELINE_JSON ===" | tee -a "$RESULTS"
fi

# ──────────────────────────────────────────────────────────────────────
# Per-cell × per-epoch evals
# ──────────────────────────────────────────────────────────────────────
for cell_dir in "$GRID_ROOT"/*/; do
  cell=$(basename "$cell_dir")
  method="${cell%%_*}"   # split on first underscore: bct_t1_sft01 → bct

  # Filter to methods of interest
  if ! [[ " $METHODS " =~ " $method " ]]; then
    continue
  fi

  echo "=== Cell: $cell (method=$method) ===" | tee -a "$RESULTS"

  for epoch in 1 2 3; do
    out_json="${cell_dir}eval_epoch_${epoch}.json"
    if [[ -f "$out_json" ]]; then
      echo "  epoch $epoch — cached ($out_json)" | tee -a "$RESULTS"
      continue
    fi

    # epoch_N* matches both `epoch_3` and `epoch_3__TIMESTAMP`; pick newest if multiple
    lora=$(ls -dt "${cell_dir}epoch_${epoch}"* 2>/dev/null | head -1)
    if [[ -z "$lora" ]]; then
      echo "  epoch $epoch — no checkpoint, skipping" | tee -a "$RESULTS"
      continue
    fi

    echo "  epoch $epoch → $lora" | tee -a "$RESULTS"
    python -m experiments.prefill.prefill_run_evals \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --checkpoint "$lora" \
        --baseline_json "$BASELINE_JSON" \
        --output_json "$out_json" \
        --limit "$HARMFUL_LIMIT" \
        --n-mmlu 0 \
        --metric-prefix "${cell}_e${epoch}/" \
        ${quant_args[@]:+"${quant_args[@]}"} \
        ${coh_args[@]:+"${coh_args[@]}"} \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        2>&1 | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
  done
done

echo "=== Done: $(date) ===" | tee -a "$RESULTS"
echo ""
echo "Per-cell eval JSONs: $GRID_ROOT/<cell>/eval_epoch_<N>.json"
echo "Run prefill_pick_best.py --model_tag $MODEL_TAG  to produce <method>_best.json"
