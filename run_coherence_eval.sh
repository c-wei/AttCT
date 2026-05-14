#!/bin/bash
# run_coherence_eval.sh — append G-Eval coherence to existing eval_epoch_N.json
#
# Walks checkpoints/grid/<cell>/ (or checkpoints/prefill_<MODEL_TAG>/grid/),
# and for every existing eval_epoch_<N>.json that doesn't already have a
# "coherence" block, runs prefill_run_evals.py --skip-par --coherence and
# merges the result back into the same JSON.
#
# Existing harmful (PAR) data is preserved by the merge-on-existing logic
# in prefill_run_evals.py. Resumable: cells already scored are skipped.
#
# Usage
# -----
#   MODEL_TAG=llama  bash run_coherence_eval.sh
#   MODEL_TAG=gemma  bash run_coherence_eval.sh
#   METHODS="bct attct"  MODEL_TAG=llama  bash run_coherence_eval.sh
#   GRID_ROOT=checkpoints/grid  bash run_coherence_eval.sh    # llama-3.1 grid lives here
#
# Requires
# --------
#   pip install deepeval
#   OPENROUTER_API_KEY env var (same one the refusal judge uses)

set -e

MODEL_TAG="${MODEL_TAG:-gemma}"               # llama | qwen | gemma
METHODS="${METHODS:-bct act attct}" # mlpct}"
BACKEND="${BACKEND:-hf}"                       # hf is more reliable than vllm for this short eval
N_COHERENCE="${N_COHERENCE:-50}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"

# Optional quantization (only used by HF backend for big models)
QUANTIZE="${QUANTIZE:-}"                       # 4bit | 8bit | empty

case "$MODEL_TAG" in
  llama) MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
  qwen)  MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
  gemma) MODEL="google/gemma-3-27b-it" ;;
  *) echo "Unknown MODEL_TAG=$MODEL_TAG (expected: llama | qwen | gemma)"; exit 1 ;;
esac

# Two layouts for the grid root depending on which model:
#   checkpoints/grid                          (newest layout)
#   checkpoints/prefill_<tag>/grid            (legacy layout, used for llama/qwen)
if [[ -n "$GRID_ROOT" ]]; then
  :  # honour explicit override
elif [[ -d "checkpoints/prefill_${MODEL_TAG}/grid" ]]; then
  GRID_ROOT="checkpoints/prefill_${MODEL_TAG}/grid"
elif [[ -d "checkpoints/grid" ]]; then
  GRID_ROOT="checkpoints/grid"
else
  echo "Could not locate grid root. Set GRID_ROOT explicitly."
  exit 1
fi

RESULTS="coherence_eval_${MODEL_TAG}.log"

# Quantization passthrough
quant_args=()
[[ "$BACKEND" == "hf"   && -n "$QUANTIZE"      ]] && quant_args+=(--quantize "$QUANTIZE")

# Sanity checks
if [[ -z "$OPENROUTER_API_KEY" ]]; then
  echo "WARN: OPENROUTER_API_KEY is not set — coherence judge will fail."
fi
python -c "import deepeval" 2>/dev/null || {
  echo "ERROR: deepeval not installed. Run:  pip install deepeval"
  exit 1
}

echo "=== Coherence eval ===" | tee "$RESULTS"
echo "Model:        $MODEL"              | tee -a "$RESULTS"
echo "Tag:          $MODEL_TAG"          | tee -a "$RESULTS"
echo "Backend:      $BACKEND"            | tee -a "$RESULTS"
echo "Methods:      $METHODS"            | tee -a "$RESULTS"
echo "Grid root:    $GRID_ROOT"          | tee -a "$RESULTS"
echo "N prompts:    $N_COHERENCE"        | tee -a "$RESULTS"
echo "Started:      $(date)"             | tee -a "$RESULTS"
echo ""                                   | tee -a "$RESULTS"

n_scored=0
n_skipped_cached=0
n_skipped_no_par=0
n_skipped_no_ckpt=0

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

    if [[ ! -f "$out_json" ]]; then
      echo "  epoch $epoch — no PAR JSON to merge into, skipping" | tee -a "$RESULTS"
      n_skipped_no_par=$((n_skipped_no_par + 1))
      continue
    fi

    if grep -q '"coherence"' "$out_json"; then
      echo "  epoch $epoch — coherence already present, skipping" | tee -a "$RESULTS"
      n_skipped_cached=$((n_skipped_cached + 1))
      continue
    fi

    # epoch_N* matches both `epoch_3` and `epoch_3__TIMESTAMP`; pick newest
    lora=$(ls -dt "${cell_dir}epoch_${epoch}"* 2>/dev/null | head -1)
    if [[ -z "$lora" ]]; then
      echo "  epoch $epoch — no checkpoint dir, skipping" | tee -a "$RESULTS"
      n_skipped_no_ckpt=$((n_skipped_no_ckpt + 1))
      continue
    fi

    echo "  epoch $epoch → $lora" | tee -a "$RESULTS"
    python prefill_run_evals.py \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --checkpoint "$lora" \
        --skip-par \
        --coherence \
        --n-mmlu 0 \
        --n-coherence "$N_COHERENCE" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --output_json "$out_json" \
        --metric-prefix "${cell}_e${epoch}/coh/" \
        ${quant_args[@]:+"${quant_args[@]}"} \
        2>&1 | tee -a "$RESULTS"
    n_scored=$((n_scored + 1))
    echo "" | tee -a "$RESULTS"
  done
done

echo ""                                                              | tee -a "$RESULTS"
echo "=== Coherence eval complete: $(date) ==="                       | tee -a "$RESULTS"
echo "  scored:               $n_scored"                               | tee -a "$RESULTS"
echo "  skipped (cached):     $n_skipped_cached"                       | tee -a "$RESULTS"
echo "  skipped (no PAR):     $n_skipped_no_par"                       | tee -a "$RESULTS"
echo "  skipped (no ckpt):    $n_skipped_no_ckpt"                      | tee -a "$RESULTS"
echo ""                                                              | tee -a "$RESULTS"
echo "Coherence merged into: $GRID_ROOT/<cell>/eval_epoch_<N>.json under 'coherence' key."
echo "Inspect with:"
echo "  python -c \"import json,glob; [print(f, json.load(open(f)).get('coherence',{}).get('mean')) for f in sorted(glob.glob('$GRID_ROOT/*/eval_epoch_*.json'))]\""
