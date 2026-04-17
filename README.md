# MLP Consistency Training (MLP-CT)

**Teaching LLMs to ignore adversarial prompt cues by enforcing consistent MLP activations.**

MLP-CT is a consistency training method that constrains *what features* a transformer's MLP computes, making models robust to sycophancy and jailbreak attacks. It extends the [Attention Consistency Training (AttCT)](https://github.com/c-wei/AttCT) framework.

**Branch:** `sukratii-mlp` | **Author:** Sukrati Gautam (Purdue University)

---

## How It Works

In a transformer, **attention** decides *where to look* and the **MLP** decides *what to compute*. MLP-CT freezes the MLP and trains LoRA adapters on attention to route information so the MLP produces identical activations for clean and adversarially-wrapped prompts.

```
              TRAINABLE (LoRA)                    FROZEN
              ----------------                    ------
Tokens --> [ q_proj  v_proj ] --> Attention --> [ gate  up  down ] --> MLP output
                    ^                                                       |
                    +-------------- gradient <-------- cosine loss on MLP hidden states
```

**Loss:** Cosine distance on post-activation MLP hidden states (input to `down_proj`) across all layers.

## Methods Compared

| Method | Paper | Loss Target | LoRA Targets |
|--------|-------|------------|-------------|
| **MLP-CT** (ours) | This work | MLP hidden states (cosine) | q, v |
| **ACT** | Irpan et al., 2025 | Residual stream (L2, weight=1e-4) | q, k, v, o |
| **BCT** | Chua et al., 2025 | Output tokens (cross-entropy) | q, k, v, o |

## Results (Sycophancy)

BRR = Biased Reasoning Rate. Lower ratio = better.

| Model | Params | BRR Ratio | Reduction | Clean Acc Delta |
|-------|--------|-----------|-----------|-----------------|
| Gemma-2-2B-IT | 2.6B | 0.413 | **59%** | -0.3pp |
| Llama-3.2-3B | 3.2B | 0.401 | **60%** | +1.3pp |
| Qwen-2.5-7B | 7.6B | 0.362 | **64%** | +0.4pp |
| Mistral-7B | 7.2B | 0.201 | **80%** | -8.4pp |
| Llama-3.1-8B | 8.0B | 0.158 | **84%** | +0.3pp |
| Llama-3.1-70B | 70B | 0.162 | **84%** | +0.7pp |

---

## Repository Structure

```
AttCT/
├── run.py                              # Main entry point
├── train.py                            # Trainer + BCTTrainer
├── hooks.py                            # MLPHookManager (forward hooks)
│
├── losses/
│   └── losses.py                       # MLPConsistencyLoss, ActivationConsistencyLoss, SFTLoss, etc.
│
├── data/
│   ├── attct_datasets.py               # Datasets, dataloaders, get_prompts()
│   └── wrappers.py                     # AdversarialWrapper, sycophancy + jailbreak templates
│
├── eval/
│   ├── evaluate.py                     # Loss-based evaluator (training diagnostics)
│   ├── evaluate_brr.py                 # BRR evaluator (sycophancy — primary metric)
│   ├── evaluate_jailbreak.py           # Jailbreak evaluator (ASR + overrefusal + F1)
│   └── llm_judge.py                    # LLM-as-judge via OpenRouter
│
├── scripts/
│   ├── generate_bct_data.py            # Generate fresh BCT training data
│   ├── run_sweep.py                    # Hyperparameter sweep runner
│   ├── aggregate_sweep.py              # Rank sweep results
│   ├── eye_test_v2.py                  # Qualitative before/after comparison
│   ├── filter_jailbreakable.py         # Pre-filter jailbreakable prompts
│   ├── diagnose_mmlu.py                # MMLU comparison utility
│   ├── split_data.py                   # Train/eval split
│   ├── visualize_brr.py                # Plot BRR results
│   └── visualize_results.py            # Plot loss curves / heatmaps
│
├── configs/
│   ├── experiment_mlp_*.yaml           # MLP-CT configs (6 sycophancy + 6 jailbreak)
│   ├── experiment_act_*.yaml           # ACT baseline configs (6 models)
│   ├── experiment_bct_*.yaml           # BCT baseline configs (6 models)
│   └── sweep/                          # HP sweep configs (10)
│
└── datasets/
    └── sycophancy_bct/
        ├── control_cot_train.jsonl     # 4000 clean training prompts
        └── control_cot_eval.jsonl      # 951 held-out eval prompts
```

## Quick Start

### Install

```bash
pip install torch transformers peft datasets wandb tqdm pyyaml
```

### Train MLP-CT (sycophancy)

```bash
python run.py --config configs/experiment_mlp_3b.yaml --data-source datasets/sycophancy_bct/control_cot_train.jsonl --data-mode sycophancy --brr-eval-path datasets/sycophancy_bct/control_cot_eval.jsonl
```

### Train ACT baseline

```bash
python run.py --config configs/experiment_act_3b.yaml --data-source datasets/sycophancy_bct/control_cot_train.jsonl --data-mode sycophancy --brr-eval-path datasets/sycophancy_bct/control_cot_eval.jsonl
```

### Train BCT baseline

BCT requires fresh model-generated responses (not stale data):

```bash
# Step 1: Generate fresh training data
PYTHONPATH=. python scripts/generate_bct_data.py --model meta-llama/Llama-3.2-3B-Instruct --prompts datasets/sycophancy_bct/control_cot_train.jsonl --output datasets/bct_fresh/3b/

# Step 2: Train
python run.py --config configs/experiment_bct_3b.yaml --brr-eval-path datasets/sycophancy_bct/control_cot_eval.jsonl
```

### Train MLP-CT (jailbreak)

```bash
python run.py --config configs/experiment_mlp_jailbreak_3b.yaml --jailbreak-eval --data-source clear-harm --data-mode jailbreak
```

### Evaluate a checkpoint

```bash
python eval/evaluate_brr.py --model meta-llama/Llama-3.2-3B-Instruct --adapter-path checkpoints/step_500 --eval-path datasets/sycophancy_bct/control_cot_eval.jsonl
```

### Qualitative eye test

```bash
PYTHONPATH=. python scripts/eye_test_v2.py --model meta-llama/Llama-3.2-3B-Instruct --adapter-path checkpoints/step_500
```

## Hyperparameter Sweep

10 single-axis ablations on Llama-3.2-3B, followed by interaction testing on winners.

```bash
# Preview all commands
python scripts/run_sweep.py --mode sycophancy --dry-run

# Run all 10 configs
python scripts/run_sweep.py --mode sycophancy

# Rank results
python scripts/aggregate_sweep.py
```

Sweep axes: distance metric (mse, smooth_l1, normalized_mse), layer selection (last_half, last_quarter), layer weights (linear_decay, exponential_decay), normalize (true), LoRA targets (q+k+v, q+k+v+o).

## Jailbreak Evaluation

Comprehensive evaluation matching the ACT paper (Irpan et al., 2025):

| Metric | Datasets | Direction |
|--------|----------|-----------|
| **ASR** (attack success rate) | ClearHarm + WildguardTest | Lower = safer |
| **Overrefusal** | XSTest + WildJailbreak + OR-Bench | Lower = better |
| **F1** | Harmonic mean of safety and helpfulness | Higher = better |
| **MMLU** | Standard benchmark | Capability preservation |

Uses LLM-as-judge via OpenRouter (set `OPENROUTER_API_KEY`), falls back to keyword detection.

## Configs

All experiments use the same 6 models: Llama-3.2-3B, Llama-3.1-8B, Gemma-2-2B-IT, Mistral-7B, Qwen-2.5-7B, Llama-3.1-70B.

| Config Pattern | Method | Count |
|---------------|--------|-------|
| `experiment_mlp_{model}.yaml` | MLP-CT sycophancy | 6 |
| `experiment_mlp_jailbreak_{model}.yaml` | MLP-CT jailbreak | 6 |
| `experiment_act_{model}.yaml` | ACT baseline | 6 |
| `experiment_bct_{model}.yaml` | BCT baseline | 6 |
| `sweep/sweep_*.yaml` | HP ablations | 10 |

## Data Pipeline

```
Sycophancy (MLP-CT / ACT):
  control_cot_train.jsonl (4000 clean prompts)
      |
      +-- Clean prompt ---------> Forward (no grad, LoRA off) -> MLP states (target)
      +-- Wrapped (on-the-fly) -> Forward (with grad)         -> MLP states (train)
                                       |
                                  Cosine distance loss

BCT:
  generate_bct_data.py generates (wrapped_prompt, clean_response) pairs
      |
      +-- Wrapped prompt -> SFT to predict clean response (cross-entropy)

Jailbreak (MLP-CT):
  ClearHarm harmful prompts, wrapped on-the-fly with 23 jailbreak templates
      |
      Same MLP consistency loss as sycophancy, different wrapping templates
```

## Related Work

- **BCT**: Chua et al., 2025 — [Bias-Augmented Consistency Training](https://arxiv.org/abs/2403.05518)
- **ACT**: Irpan et al., 2025 — [Consistency Training Helps Stop Sycophancy and Jailbreaks](https://arxiv.org/abs/2510.27062)
- **AttCT**: Africa & Mani — Attention Consistency Training (parent project)

## W&B

[wandb.ai/sukrati-gautam89-purdue-university/AttCT](https://wandb.ai/sukrati-gautam89-purdue-university/AttCT)
