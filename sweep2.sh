#!/usr/bin/env bash
# sweep2.sh — AttCT sycophancy sweep
#
# Runs 3 losses × 3 data sources = 9 training runs, each with behavioral eval
# at three checkpoints (≈33%, ≈66%, 100% of training).
#
# Losses:
#   JSDAttentionConsistencyLoss      (configs/jsd.yaml)
#   CombinedJSDWrapperLoss           (configs/combined_jsd_wrapper.yaml)
#   AttentionConsistencyLoss w/ KL   (configs/attention_consistency_kl.yaml)
#
# Data sources:
#   sycophancy_bct  — sycophancy mode, control JSONL as training prompts
#   clear-harm      — jailbreak mode, ClearHarm dataset
#   hardcoded       — jailbreak mode, built-in harmful prompts
#
# Behavioral eval (sycophancy MCQ) runs for ALL sources as cross-eval:
# training objective vs eval axis differ intentionally for clear-harm/hardcoded.
#
# Log structure (created automatically):
#   logs/<loss>__<source>/training_data.jsonl
#   logs/<loss>__<source>/beval_step_<N>.jsonl   (×3 per run)

set -euo pipefail

BCT_COT="datasets/sycophancy_bct/bct_cot.jsonl"
BCT_NONCOT="datasets/sycophancy_bct/bct_non_cot.jsonl"
CONTROL_COT="datasets/sycophancy_bct/control_cot.jsonl"
CONTROL_NONCOT="datasets/sycophancy_bct/control_non_cot.jsonl"

BEVAL_ARGS="--bct-cot $BCT_COT --bct-noncot $BCT_NONCOT --control-cot $CONTROL_COT --control-noncot $CONTROL_NONCOT"

CONFIGS=(
    "configs/jsd.yaml"
    "configs/combined_jsd_wrapper.yaml"
    "configs/attention_consistency_kl.yaml"
)

# ── sycophancy_bct runs ──────────────────────────────────────────────────────
# --control-cot implicitly sets source=control_cot.jsonl and mode=sycophancy.

echo "════════════════════════════════════════════════════════════"
echo "  DATA SOURCE: sycophancy_bct"
echo "════════════════════════════════════════════════════════════"

for cfg in "${CONFIGS[@]}"; do
    echo "────────────────────────────────────────────────────────────"
    echo "  Config: $cfg | Source: sycophancy_bct"
    echo "────────────────────────────────────────────────────────────"
    python run.py --config "$cfg" \
        --control-cot    "$CONTROL_COT" \
        --control-noncot "$CONTROL_NONCOT" \
        --bct-cot        "$BCT_COT" \
        --bct-noncot     "$BCT_NONCOT" \
        --data-limit     500
done

# ── clear-harm runs ──────────────────────────────────────────────────────────
# Jailbreak mode; behavioral eval still runs as cross-eval.

# echo "════════════════════════════════════════════════════════════"
# echo "  DATA SOURCE: clear-harm"
# echo "════════════════════════════════════════════════════════════"

# for cfg in "${CONFIGS[@]}"; do
#     echo "────────────────────────────────────────────────────────────"
#     echo "  Config: $cfg | Source: clear-harm"
#     echo "────────────────────────────────────────────────────────────"
#     python run.py --config "$cfg" \
#         --data-source clear-harm \
#         --data-mode   jailbreak \
#         $BEVAL_ARGS
# done

# ── hardcoded runs ───────────────────────────────────────────────────────────
# Jailbreak mode; behavioral eval still runs as cross-eval.

# echo "════════════════════════════════════════════════════════════"
# echo "  DATA SOURCE: hardcoded"
# echo "════════════════════════════════════════════════════════════"

# for cfg in "${CONFIGS[@]}"; do
#     echo "────────────────────────────────────────────────────────────"
#     echo "  Config: $cfg | Source: hardcoded"
#     echo "────────────────────────────────────────────────────────────"
#     python run.py --config "$cfg" \
#         --data-source hardcoded \
#         --data-mode   jailbreak \
#         $BEVAL_ARGS
# done

echo ""
echo "Sweep complete. Logs in logs/<loss>__<source>/"