#!/usr/bin/env bash
set -euo pipefail

COMMON="--model llama \
  --data-mode sycophancy \
  --data-source sycophancy_bct \
  --interleave \
  --kl-dataset ultrachat \
  --max-steps 5000 \
  --eval-sycophancy \
  --wandb-group sycophancy-ablations \
  --save-dir checkpoints \
  --hf-repo anon/sweeps_attct"

# 1. Baseline — JSD, q+v, uniform, alpha=16
python run.py --config configs/jsd.yaml $COMMON \
  --run-name ablation_baseline

# 2. LoRA q+k+v+o — all attention projections
python run.py --config configs/jsd.yaml $COMMON \
  --run-name ablation_lora_qkvo \
  --lora-targets q_proj k_proj v_proj o_proj

# 3. LoRA q+k+v — without o_proj
python run.py --config configs/jsd.yaml $COMMON \
  --run-name ablation_lora_qkv \
  --lora-targets q_proj k_proj v_proj

# 4. LoRA q+k+v+o + exponential layer weight decay
python run.py --config configs/jsd_exp.yaml $COMMON \
  --run-name ablation_lora_qkvo_exp \
  --lora-targets q_proj k_proj v_proj o_proj

# 5. Exponential layer weight decay — q+v only
python run.py --config configs/jsd_exp.yaml $COMMON \
  --run-name ablation_exp_decay

# 6. Alpha=8 (LoRA scale = 1.0)
python run.py --config configs/jsd.yaml $COMMON \
  --run-name ablation_alpha_8 \
  --lora-alpha 8

# 7. Alpha=32 (LoRA scale = 4.0)
python run.py --config configs/jsd.yaml $COMMON \
  --run-name ablation_alpha_32 \
  --lora-alpha 32

# 8. Last half layers — JSD on layers 16-31 only
python run.py --config configs/jsd.yaml $COMMON \
  --run-name ablation_last_half_layers \
  --loss-layer-selection last_half

# 9. Last quarter layers — JSD on layers 24-31 only
python run.py --config configs/jsd.yaml $COMMON \
  --run-name ablation_last_quarter_layers \
  --loss-layer-selection last_quarter
