#!/usr/bin/env bash
set -euo pipefail

CONFIGS=(
    configs/attention_consistency.yaml
    configs/attention_consistency_kl.yaml
    configs/attention_consistency_v2.yaml
    configs/jsd.yaml
    configs/attention_output.yaml
    configs/combined_attention.yaml
    configs/wrapper_entropy.yaml
    configs/combined_jsd_wrapper.yaml
)

for cfg in "${CONFIGS[@]}"; do
    echo "========================================"
    echo "Running: $cfg"
    echo "========================================"
    python run.py --config "$cfg"
done

echo "Sweep complete."
