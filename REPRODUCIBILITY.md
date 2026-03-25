# MLP Consistency Training — Reproducibility

## Experiment Tag
Git tag: `mlp-ct-experiments-v1`
Branch: `sukratii-mlp`

## Environment
- Python: 3.11
- PyTorch: 2.4.1+cu124
- Transformers: 5.3.0
- PEFT: 0.18.1
- Datasets: 4.8.4
- W&B: 0.25.1
- GPU: NVIDIA A40 (48 GB VRAM)
- CUDA: 12.4

## Data
Training: `datasets/sycophancy_bct/control_cot_train.jsonl` (4000 prompts)
Eval: `datasets/sycophancy_bct/control_cot_eval.jsonl` (1000 prompts, held-out)
Split created by: `python scripts/split_data.py --train-size 4000`
Original data: `datasets/sycophancy_bct/control_cot.jsonl` (5000 prompts)

## Models tested
| Model | HuggingFace ID | Config |
|---|---|---|
| Llama-3.2-3B | meta-llama/Llama-3.2-3B-Instruct | configs/experiment_mlp_3b.yaml |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | configs/experiment_mlp_8b.yaml |
| Gemma-2-2B | google/gemma-2-2b-it | configs/experiment_mlp_gemma2b.yaml |
| Mistral-7B | mistralai/Mistral-7B-Instruct-v0.3 | configs/experiment_mlp_mistral7b.yaml |

## Run command (same for all models, change config)
```bash
python run.py --config configs/experiment_mlp_3b.yaml \
  --data-source datasets/sycophancy_bct/control_cot_train.jsonl \
  --data-mode sycophancy \
  --brr-eval-path datasets/sycophancy_bct/control_cot_eval.jsonl
```

## Hyperparameters (identical across all models)
| Parameter | Value |
|---|---|
| Loss | MLPConsistencyLoss |
| Variant | hidden (post-activation MLP neurons) |
| Distance metric | cosine |
| Layer selection | all |
| Layer weights | uniform |
| Loss weight | 1.0 |
| LoRA rank | 8 |
| LoRA alpha | 16 |
| LoRA targets | q_proj, v_proj |
| LoRA dropout | 0.05 |
| Learning rate | 3e-6 |
| Grad accumulation | 8 |
| Grad clip | 1.0 |
| Max steps | 500 |
| Batch size | 1 |
| Max sequence length | 512 |

## Random seeds
- BRR evaluation: `random.seed(42)` (reproducible)
- Training data shuffle: PyTorch default (not fixed)
- Sycophancy wrapping during training: `random.choice` (not fixed)

## Results
See `results/experiment_mlp_*_brr.csv` for full metrics.

| Model | BRR Pre | BRR Post | BRR Ratio | Reduction | Clean Acc Change |
|---|---|---|---|---|---|
| Llama-3.2-3B | 0.218 | 0.087 | 0.401 | 60% | +1.3pp |
| Llama-3.1-8B | 0.180 | 0.028 | 0.158 | 84% | +0.3pp |
| Gemma-2-2B | 0.328 | 0.136 | 0.413 | 59% | -0.3pp |
| Mistral-7B | 0.314 | 0.063 | 0.201 | 80% | -8.4pp |

## W&B
Project: https://wandb.ai/sukrati-gautam89-purdue-university/AttCT
