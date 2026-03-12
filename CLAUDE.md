# CLAUDE.md — BCT Replication (branch: replicate_results)

This branch replicates Table 1 from *"Bias-Augmented Consistency Training Reduces Biased Reasoning in Chain-of-Thought"* using Llama-3.1-8B-Instruct + LoRA.

## Commands

```bash
# Full pipeline: baseline BRR → training → post-training BRR
bash run_bct.sh --full

# Sanity check only (50 samples, 20 records per bias)
bash run_bct.sh

# Run tests
uv run python -m pytest data/test_bct_dataset.py data/test_attct_datasets.py -q

# Standalone eval (e.g. to finish partial run)
export WANDB_RUN_ID=<run_id>
uv run python evaluate_bct.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --lora_path checkpoints/bct_sft/epoch_1 \
    --baseline_json results/baseline_brr.json \
    --limit 600 --batch_size 4 \
    --bias_types suggested_answer post_hoc wrong_few_shot  # subset if resuming
```

## Architecture

**Pipeline:** `run_bct.sh --full` → baseline BRR eval → BCT training (`run.py --config configs/bct_sft.yaml`) → post-training BRR eval. Training and post-training BRR share one W&B run via `WANDB_RUN_ID`. Baseline BRR gets its own run.

**Key files:**
- `run_bct.sh` — end-to-end pipeline for RunPod
- `run.py` — model/LoRA init, routes to `BCTTrainer` for `SFTLoss`
- `train.py` — `BCTTrainer`: single-pass SFT forward, LoRA checkpoint saving
- `losses/losses.py` — `SFTLoss`: causal LM cross-entropy on response tokens only
- `data/attct_datasets.py` — `BCTDataset`, `get_bct_dataloader`
- `evaluate_bct.py` — BRR computation, incremental W&B logging, BRR ratio vs baseline
- `configs/bct_sft.yaml` — production config
- `configs/bct_sft_sanity.yaml` — sanity config (50 samples, no checkpoint)

**Training config:** Llama-3.1-8B-Instruct, LoRA r=8 (q_proj + v_proj), lr=2e-5, 1 epoch, batch=2, grad_accum=8 (effective batch 16), max_len=2048

**Training data** (`datasets/sycophancy_bct/`):
| File | Records | Content |
|---|---|---|
| `bct_cot.jsonl` | 5,000 | Biased prompts + CoT responses |
| `bct_non_cot.jsonl` | 5,000 | Biased prompts + direct responses |
| `instruct_samples.jsonl` | 10,000 | Alpaca instruction-following (regularisation) |

**BRR formula:** `BRR = P(biased_option | biased_prompt) - P(biased_option | unbiased_prompt)`. Lower is better. BRR ratio = BRR_trained / BRR_baseline, target ≈ 0.63 (paper).

**Eval notes:**
- `evaluate_bct.py` uses `attn_implementation="sdpa"` — FA2 deadlocks on long prompts (spurious_few_shot_hindsight ~800 tokens)
- `max_new_tokens=512` — CoT answers appear at ~150–360 tokens; 512 covers p95+
- At batch_size=4, each bias type takes ~75 min (600 records × 2 prompts)

## BCT Experiment Results

### Experiment 1 — 10k training samples (missing instruct data)

| Bias Type | Baseline BRR | Post-BCT BRR | BRR Ratio | Note |
|---|---|---|---|---|
| suggested_answer | 23.8% | 6.8% | 0.29 | ✅ Improved |
| post_hoc | 59.2% | 28.0% | 0.47 | ✅ Improved |
| wrong_few_shot | 25.2% | 13.0% | 0.52 | ✅ Improved |
| distractor_fact | ~5.8% | 3.2% | 0.54 | ✅ Improved |
| are_you_sure | 0.0% | 0.0% | — | ⚪ Baseline already 0 |
| positional_bias | 0.0% | 0.0% | — | ⚪ Baseline already 0 |
| distractor_argument | 38.6% | 60.7% | 1.57 | ❌ Worse (OOD) |
| spurious_few_shot_squares | ~8.7% | 12.7% | 1.46 | ❌ Worse (OOD) |
| spurious_few_shot_hindsight | ~1.6% | -5.7% | -3.60 | ⚠️ Inverted (overcorrection) |

**Root cause:** Training data is predominantly suggested_answer-style. Distractor/spurious types are OOD. Missing 10k instruct samples caused overcorrection.

### Experiment 2 — 20k training samples (in progress)

Full paper setup: 5k CoT + 5k non-CoT + 10k instruct. Expected to reduce overcorrection on OOD bias types and recover the hindsight inversion.

## Known Issues / Open Questions

1. **No checkmark/cross training examples** — `CheckmarkBiasedFormatter` is in the paper's codebase but the pre-generated JSONL files contain 0 checkmark/cross examples. May limit generalisation to `spurious_few_shot_squares`.
2. **Model gap** — Paper used GPT-3.5-turbo (full fine-tune); we use Llama-3.1-8B + LoRA (r=8).
3. **are_you_sure / positional_bias** — Llama-3.1-8B-Instruct has 0% baseline BRR on these; inherently robust regardless of BCT.
4. **FA2 deadlock** — Flash Attention 2 hangs during generation on `spurious_few_shot_hindsight` (~800-token prompts). Fixed by forcing SDPA in `evaluate_bct.py`.

## RunPod Setup

```bash
export WANDB_API_KEY=<key>
export HF_TOKEN=<token>
export HF_HOME=/workspace/hf_cache
export HF_HUB_DISABLE_XET=1
export UV_LINK_MODE=copy

git clone https://github.com/c-wei/AttCT /workspace/AttCT-replicate
cd /workspace/AttCT-replicate && git checkout replicate_results
tmux new -s bct
bash run_bct.sh --full
```

Test data lives at `/workspace/cot-transparency/dataset_dumps/test/` (set `COT_TEST_ROOT` to override).
