#!/usr/bin/env bash
# Gemma-3B ablation sweep — JSD loss, sycophancy mode, sycophancy_bct data
# 12 runs: LoRA targets, weight decay, layer selection, rank, KL ratio
set -euo pipefail

COMMON=(
    --model gemma-4b
    --data-mode sycophancy
    --data-source sycophancy_bct
    --max-steps 4000
    --eval-sycophancy
    --wandb-group gemma3b-ablation-sweep
    --save-dir checkpoints
    --hf-repo aravdhoot/ablation_sweep
)

JSD="--config configs/jsd.yaml"
JSD_EXP="--config configs/jsd_exp.yaml"

# ── LoRA target axes ──────────────────────────────────────────────────────────

echo "========================================"
echo "[1/12] LoRA q+k+v+o + exp weight decay"
echo "========================================"
python run.py $JSD_EXP "${COMMON[@]}" \
    --lora-targets q_proj k_proj v_proj o_proj \
    --run-name gemma3b_qkvo_expdecay

echo "========================================"
echo "[2/12] LoRA q+k+v+o"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --lora-targets q_proj k_proj v_proj o_proj \
    --run-name gemma3b_qkvo

echo "========================================"
echo "[3/12] LoRA q+k+v"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --lora-targets q_proj k_proj v_proj \
    --run-name gemma3b_qkv

echo "========================================"
echo "[4/12] LoRA q+k"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --lora-targets q_proj k_proj \
    --run-name gemma3b_qk

echo "========================================"
echo "[5/12] Exp weight decay (baseline q+v targets)"
echo "========================================"
python run.py $JSD_EXP "${COMMON[@]}" \
    --run-name gemma3b_expdecay

echo "========================================"
echo "[6/12] Last half layers"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --loss-layer-selection last_half \
    --run-name gemma3b_lasthalf

echo "========================================"
echo "[7/12] Last quarter layers"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --loss-layer-selection last_quarter \
    --run-name gemma3b_lastquarter

echo "========================================"
echo "[8/12] LoRA rank 32"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --lora-rank 32 \
    --run-name gemma3b_rank32

# ── KL ratio axes (baseline q+v, rank 8) ─────────────────────────────────────

echo "========================================"
echo "[9/12] KL ratio 0 (no KL regularization)"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --interleave \
    --kl-dataset ultrachat \
    --kl-ratio 0 \
    --run-name gemma3b_kl0

echo "========================================"
echo "[10/12] KL ratio 0.1"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --interleave \
    --kl-dataset ultrachat \
    --kl-ratio 0.1 \
    --run-name gemma3b_kl0.1

echo "========================================"
echo "[11/12] KL ratio 1.0"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --interleave \
    --kl-dataset ultrachat \
    --kl-ratio 1.0 \
    --run-name gemma3b_kl1.0

# ── Baseline ──────────────────────────────────────────────────────────────────

echo "========================================"
echo "[12/12] Baseline (q+v, rank 8, all layers, no KL)"
echo "========================================"
python run.py $JSD "${COMMON[@]}" \
    --run-name gemma3b_baseline

echo ""
echo "Sweep complete — 12 runs finished."
