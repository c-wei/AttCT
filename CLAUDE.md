# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AttCT** (Attention Consistency Training) trains language models to maintain consistent internal attention patterns between clean and adversarially-wrapped prompts, defending against jailbreak and sycophancy attacks via LoRA fine-tuning.

## Commands

### Setup
```bash
uv sync          # Install dependencies via uv (preferred)
pip install -e . # Or standard pip
```

### Training
```bash
python run.py                                          # Default config
python run.py --config configs/attention_consistency_kl.yaml  # Specific config
bash sweep.sh                                          # Run all 8 main configs
```

### Testing
```bash
pytest data/test_attct_datasets.py  # Data pipeline tests
pytest data/test_wrappers.py        # Adversarial wrapper tests
```

### Sanity check (tiny-gpt2, fast)
```bash
python run.py --config configs/sanity_bct.yaml
python run.py --config configs/sanity_act.yaml
```

## Architecture

### Training pipeline
`run.py` loads a YAML config, instantiates the HF model with a PEFT LoRA wrapper, selects a loss from `LOSS_REGISTRY`, initializes wandb, then runs `Trainer` → `Evaluator`.

### Two-pass forward
Most losses require **two forward passes** per step: one on the clean prompt, one on the adversarially-wrapped prompt. `Trainer._step()` coordinates both and passes outputs to the loss function. `WrapperEntropyRegularizationLoss` is the exception (no clean pass needed).

### Loss functions (`losses/losses.py`)
All losses inherit `ConsistencyLoss(nn.Module)` and implement `forward(clean_outputs, adv_outputs, start_index, clean_len)`. The `start_index` and `clean_len` fields are used to slice the attention/hidden-state tensors so only the clean-region tokens are compared across the two passes.

Loss classes in `LOSS_REGISTRY`:
- **AttentionConsistencyLoss** — L2 or KL on per-layer attention weights
- **AttentionConsistencyLossV2** — head-averaged attention comparison
- **JSDAttentionConsistencyLoss** — Jensen-Shannon divergence (symmetric, bounded)
- **AttentionOutputConsistencyLoss** — match attention-weighted hidden states
- **CombinedAttentionConsistencyLoss** — KL on attention + MSE on hidden states
- **ActivationConsistencyLoss** — residual stream activations
- **BehavioralConsistencyLoss** — output logit KL/MSE/CE (no clean pass needed)
- **WrapperEntropyRegularizationLoss** — suppresses attention to wrapper tokens (halves memory)
- **CombinedJSDWrapperLoss** — JSD toward clean + entropy suppression on wrapper

### Data pipeline (`data/`)
`AttCTDataset` takes a list of clean prompts and for each generates a `(clean, wrapped)` pair. `get_dataloader()` is the main entry point; `get_prompts()` handles data sources.

**Supported data sources** (`data.source` in config):
- `"clear-harm"` — ClearHarm HF dataset
- `"hardcoded"` — 10 built-in harmful prompts
- `"sycophancy_bct"` — local JSONL files under `datasets/sycophancy_bct/`
- `"<file_path>"` — custom JSONL/TXT file

**Modes** (`data.mode`):
- `"jailbreak"` — uses jailbreak prompt templates; output keys: `wrapped_input_ids`
- `"sycophancy"` — uses sycophancy templates; output keys: `adv_input_ids` + `wrapper_mask`

### Batch format (batch_size=1 enforced)
```python
{
    'clean_input_ids': Tensor[1, clean_seq_len],
    'clean_attention_mask': Tensor[1, clean_seq_len],
    'wrapped_input_ids': Tensor[1, wrapped_seq_len],   # or 'adv_input_ids'
    'wrapped_attention_mask': Tensor[1, wrapped_seq_len],
    'start_index': Tensor[1],   # where clean prompt starts inside wrapped
    'clean_len': Tensor[1],     # token length of clean prompt
    'wrapper_mask': Tensor[1, wrapped_seq_len],  # sycophancy mode only
}
```
`start_index` and `clean_len` must be uniform within a batch — validated in both `Trainer` and `Evaluator`.

### Config structure
YAML configs live in `configs/`. `run.py` deep-merges any specified config on top of defaults. Key sections: `model`, `lora`, `training`, `loss` (name + kwargs), `data`.

## Environment Variables
- `HF_TOKEN` — for gated HuggingFace models (e.g. Llama)
- `OPENROUTER_API_KEY` — only needed for `icl_hitler_experiment.py`
- `WANDB_API_KEY` — for wandb logging
