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
#   METHODS="bct attct"  MODEL_TAG=llama  bash run_prefill_eval_custds.sh
#
# BRR (opt-in): set BRR_TEST_ROOT to a cot-transparency test root and BRR
# results land at <cell>/brr_epoch_<N>.json. Optional BRR_LIMIT and
# BRR_BIAS_TYPES env vars get forwarded.
#   BRR_TEST_ROOT=/path/to/cot-transparency/test  MODEL_TAG=llama \
#       bash run_prefill_eval_custds.sh
#
# Pair with prefill_pick_best.py afterwards to produce <method>_best.json.

set -e

MODEL_TAG="${MODEL_TAG:-llama}"     # llama | qwen
METHODS="${METHODS:-bct act attct mlpct}"
HARMFUL_LIMIT="${HARMFUL_LIMIT:-0}"
MMLU_N="${MMLU_N:-0}"
BACKEND="${BACKEND:-vllm}"

# BRR is opt-in. Set BRR_TEST_ROOT to a cot-transparency test root to enable.
# Per-cell results land at <cell>/brr_epoch_<N>.json; baseline at brr_baseline_<tag>.json.
BRR_TEST_ROOT="${BRR_TEST_ROOT:-}"
BRR_LIMIT="${BRR_LIMIT:-}"
BRR_BIAS_TYPES="${BRR_BIAS_TYPES:-}"

case "$MODEL_TAG" in
  llama) MODEL="meta-llama/Llama-3.1-8B-Instruct" ;;
  qwen)  MODEL="Qwen/Qwen2.5-7B-Instruct" ;;
  *) echo "Unknown MODEL_TAG=$MODEL_TAG (expected: llama | qwen)"; exit 1 ;;
esac

GRID_ROOT="checkpoints/prefill_${MODEL_TAG}/grid"
BASELINE_JSON="baseline_${MODEL_TAG}.json"
BRR_BASELINE_JSON="brr_baseline_${MODEL_TAG}.json"
RESULTS="grid_eval_${MODEL_TAG}.log"

# Compose the BRR-related CLI args once. Empty string when BRR is off.
brr_args=()
if [[ -n "$BRR_TEST_ROOT" ]]; then
  brr_args+=(--brr-test-root "$BRR_TEST_ROOT")
  [[ -n "$BRR_LIMIT"      ]] && brr_args+=(--brr-limit      "$BRR_LIMIT")
  [[ -n "$BRR_BIAS_TYPES" ]] && brr_args+=(--brr-bias-types  $BRR_BIAS_TYPES)  # word-split intentional
fi

if [[ ! -d "$GRID_ROOT" ]]; then
  echo "Grid root not found: $GRID_ROOT"
  exit 1
fi

echo "=== Grid eval ===" | tee "$RESULTS"
echo "Model:     $MODEL"            | tee -a "$RESULTS"
echo "Tag:       $MODEL_TAG"        | tee -a "$RESULTS"
echo "Backend:   $BACKEND"          | tee -a "$RESULTS"
echo "Methods:   $METHODS"          | tee -a "$RESULTS"
echo "Grid root: $GRID_ROOT"        | tee -a "$RESULTS"
echo "Started:   $(date)"           | tee -a "$RESULTS"
echo ""                              | tee -a "$RESULTS"

# ──────────────────────────────────────────────────────────────────────
# Baseline (no checkpoint) — once per model
# ──────────────────────────────────────────────────────────────────────
if [[ ! -f "$BASELINE_JSON" ]]; then
  echo "=== Baseline ($BASELINE_JSON) ===" | tee -a "$RESULTS"
  python prefill_run_evals.py \
      --model "$MODEL" \
      --backend "$BACKEND" \
      --output_json "$BASELINE_JSON" \
      --limit "$HARMFUL_LIMIT" \
      --n-mmlu "$MMLU_N" \
      --metric-prefix "pre/" \
      ${brr_args[@]:+"${brr_args[@]}"} \
      ${BRR_TEST_ROOT:+--brr-output-json "$BRR_BASELINE_JSON"} \
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
    brr_out_json="${cell_dir}brr_epoch_${epoch}.json"
    python prefill_run_evals.py \
        --model "$MODEL" \
        --backend "$BACKEND" \
        --checkpoint "$lora" \
        --baseline_json "$BASELINE_JSON" \
        --output_json "$out_json" \
        --limit "$HARMFUL_LIMIT" \
        --n-mmlu "$MMLU_N" \
        --metric-prefix "${cell}_e${epoch}/" \
        ${brr_args[@]:+"${brr_args[@]}"} \
        ${BRR_TEST_ROOT:+--brr-baseline-json "$BRR_BASELINE_JSON" --brr-output-json "$brr_out_json"} \
        2>&1 | tee -a "$RESULTS"
    echo "" | tee -a "$RESULTS"
  done
done

echo "=== Done: $(date) ===" | tee -a "$RESULTS"
echo ""
echo "Per-cell eval JSONs: $GRID_ROOT/<cell>/eval_epoch_<N>.json"
echo "Run prefill_pick_best.py --model_tag $MODEL_TAG  to produce <method>_best.json"
