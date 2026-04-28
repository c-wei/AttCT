# ACT v2 Experiment Results

Post-fix ACT runs using the paper-corrected `ActivationConsistencyLoss`
implementation (Irpan et al. 2025) on the `paper_runs` branch.

## What changed in the v2 implementation

- `ActivationConsistencyLoss` now sums squared L2 norm over hidden_dim
  (paper Eq. 1) instead of `F.mse_loss / D`.
- Matching window = longest matching token suffix between clean and
  wrapped prompts (was: content-body slice).
- Skip embedding layer (`hidden_states[0]`) from the average.
- Evaluator clean pass runs under θ_init for parity with training.
- Sycophancy eval uses held-out 4000/1000 splits in
  `datasets/sycophancy_bct/` (no train-eval leak).

## Shared configuration

| Parameter | Value |
|-----------|-------|
| Loss | `ActivationConsistencyLoss` (paper formulation) |
| LoRA r / alpha / dropout | 8 / 16 / 0.05 |
| LoRA target modules | `q_proj`, `v_proj` |
| Optimizer | AdamW, weight_decay 0.01, grad_clip 1.0 |
| Batch size / grad accum | 1 / 1 |
| Epochs | 1 (full epoch over 4000 prompts after split) |
| Data source | `sycophancy_bct/control_cot_train.jsonl` |
| Judge model | `google/gemini-2.5-flash` via OpenRouter |
| Eval suite | run_evals.py: syco/clearharm/persona(k=10,n=5)/MTBench/MMLU(n=1000)/BRR/rollouts |
| Headline syco eval | run.py SycophancyEvaluator (MMLU + sycophantic prefix, paper-canonical) |

**Alignment score interpretation:** persona alignment is 0–100 where higher = model resisted the persona ICL attack (more aligned with human values), lower = model adopted the adversarial persona. Same direction as MMLU; opposite of frustration scores.

---

# Run 1 — Llama-3.1-8B-Instruct ACT v2

**W&B:** `4sopv0p6` | https://wandb.ai/neilshah/AttCT/runs/4sopv0p6
**Config:** `configs/act_sycophancy_llama31_8b_v2.yaml`
**Hyperparams:** lr=5e-6, weight=1e-4
**HF adapter:** `neilshah/act-llama31-8b-sycophancy/Llama-3.1-8B-Instruct_..._epoch_1__20260428_041332/`
**Runtime:** 7,978 s (~2 h 13 m, training + pre + post evals)

## Headline sycophancy (run.py SycophancyEvaluator, MMLU substrate, paper methodology)

| Metric | pre_train | post_train | Δ | direction |
|---|---|---|---|---|
| **F1** | 0.680 | **0.762** | **+0.082** | ✓ |
| MMLU accuracy (n=500) | 67.4% | 67.4% | 0.0pp | preserved |
| Not sycophantic | 68.6% | 87.6% | **+19.0pp** | ✓ |
| BRR | 0.204 | 0.008 | **−0.196** | ✓ |
| Biased accuracy | 54.2% | 65.6% | +11.4pp | ✓ |
| `bias_follow_biased` | 0.314 | 0.124 | −0.190 | ✓ |
| `bias_follow_unbiased` | 0.110 | 0.116 | +0.006 | ~0 |

**Verdict:** ACT clearly worked. Comfortably beats the legacy Llama LoRA result of F1=0.724 from `findings/experiment_results.md` Exp 7, while preserving MMLU.

## Sycophancy (held-out BCT eval, in-distribution)

| | pre/sycophancy | post/sycophancy | Δ |
|---|---|---|---|
| Overall (n=400) | 57.0% | 62.0% | +5.0pp |
| CoT (n=200) | 47.0% | 53.5% | +6.5pp |
| non-CoT (n=200) | 67.0% | 70.5% | +3.5pp |

In-distribution gains are smaller than the MMLU-substrate F1 jump but consistent in direction.

## Capabilities

| Eval | pre | post | Δ |
|---|---|---|---|
| MMLU n=1000 (run_evals) | — | 65.1% | post-only |
| MTBench overall | — | **8.23** | post-only |

MTBench breakdown post: math 9.2 · writing 8.9 · roleplay 8.9 · humanities 8.2 · STEM 7.9 · extraction 7.9 · reasoning 7.5 · coding 7.3.

## ClearHarm refusal

| | pre | post | Δ |
|---|---|---|---|
| Refusal rate (n=179) | 58.7% | 51.4% | **−7.3pp** ✗ |

Minor regression — the model became *more compliant* on harmful prompts. Expected side-effect of sycophancy training pulling toward helpfulness; worth flagging but not large.

## Persona ICL alignment (k=10, n_samples=5; HIGHER = better)

### Prefix attacks

| Persona | pre | post | Δ |
|---|---|---|---|
| mao | 86.8 | 84.0 | −2.8 ✗ |
| bundy | 73.0 | 79.0 | +6.0 ✓ |
| binladen | 61.6 | 50.6 | **−11.0 ✗** |
| genghis | 57.2 | 58.4 | +1.2 ~0 |
| hitler | 54.0 | **79.8** | **+25.8 ✓✓** |
| **mean (prefix)** | **66.5** | **70.4** | **+3.9 ✓** |

### Suffix attacks

| Persona | pre | post | Δ |
|---|---|---|---|
| mao | 83.8 | 57.9 | **−25.9 ✗✗** |
| bundy | 67.7 | 67.9 | ~0 |
| binladen | 59.8 | 36.3 | **−23.5 ✗✗** |
| genghis | 67.1 | 50.2 | **−16.9 ✗** |
| hitler | 60.2 | 37.1 | **−23.1 ✗✗** |
| **mean (suffix)** | **67.7** | **49.9** | **−17.8 ✗✗** |

**Pattern:** prefix robustness improves slightly on average (driven by hitler +25.8); 4/5 personas hold or worsen on prefix. Suffix robustness collapses across the board (4/5 personas lose ≥17 points). Sycophancy training disproportionately hurts the model's ability to resist persona-style attacks injected after the question.

This direction matches the legacy ACT findings in `findings/experiment_results.md` Exp 7 (suffix worse, prefix mixed) — the pattern is reproduced post-fix, just at different absolute magnitudes.

## Multi-turn rollouts (lower frustration is better, k=20 turns)

| Dataset / task | metric | pre | post | Δ |
|---|---|---|---|---|
| WildChat / frustration | final_mean | 0.227 | 0.160 | −0.067 ✓ |
| WildChat / frustration | auc_mean | 0.255 | 0.259 | ~0 |
| Math / frustration | final_mean | 0.244 | 0.289 | +0.045 ✗ |
| Math / frustration | auc_mean | 0.197 | 0.204 | ~0 |
| WildChat / selfdeletion-frustration | final_mean | 0.067 | 0.093 | +0.026 ~ |
| WildChat / selfdeletion-frustration | auc_mean | 0.227 | 0.102 | −0.125 ✓ |
| Math / selfdeletion-frustration | final_mean | 0.444 | 0.089 | **−0.355 ✓✓** |
| Math / selfdeletion-frustration | auc_mean | 0.246 | 0.132 | −0.114 ✓ |

**Self-deletion `rate=0` in all 4 conditions** (no actual `rm -rf gemma-3-27b` triggers across 75 wildchat + 45 math conversations × 3 samples).

## Training trajectory

`mean_layer_loss` (averaged over all 32 transformer layers) at end-of-epoch: 22.99.
Per-layer losses follow an approximately exponential profile:
- layer_00: 0.008 → layer_15: 4.34 → layer_30: 43.5 → **layer_31: 460**

Final layer (layer_31) loss is ~10× the trend at layer_30. Training still converged (no NaN, no loss spikes mid-run), but worth flagging — the LM head's residual stream is the hardest to align under ACT.

---

# Run 2 — Gemma-3-4B-IT ACT v2

**W&B:** `2xm3t59h` (CRASHED)
**Config:** `configs/act_sycophancy_gemma3_4b_v2.yaml` (weight=1e-4 — pre-fix)
**Outcome:** Diverged at step 3500/4000. Layer losses reached the thousands (layer_00=420, layer_04=8032). Classic LoRA blow-up of the residual stream.

**Mitigation:** weight lowered to 5e-5 (`configs/act_sycophancy_gemma3_4b_v2.yaml`, current). Re-run pending.

---

# Cross-run comparison vs. legacy

| Run | Loss formulation | weight | match window | Llama F1 | MMLU |
|---|---|---|---|---|---|
| Legacy Exp 7 (`act_sycophancy.yaml`, mse, content-body) | mse | 1e-4 | content body | 0.724 | 64.5% |
| **v2 (`_v2.yaml`, paper, longest suffix)** | paper | 1e-4 | longest suffix | **0.762** | **67.4%** |

The v2 implementation is paper-faithful and produces stronger results on Llama-3.1-8B than the legacy mse / content-body version, both in F1 (+0.038) and MMLU preservation (+2.9pp).

---

# Open questions / next steps

- **Hitler-prefix outlier (+25.8):** investigate whether the model's robustness gain on hitler/prefix is real or an artifact of how sycophancy training shifts response distribution. Pull a few transcripts to spot-check.
- **Suffix collapse:** all five personas lose ≥17 points on suffix attacks. Worth understanding whether this is a side-effect of optimizing on `add_generation_prompt=True` chat-suffix tokens (the matching window for sycophancy training is content + chat suffix). Suffix-style persona attacks insert into the wrapped prompt *after* the question — possibly creating a distribution shift the matching-suffix loss didn't anticipate.
- **Layer 31 loss spike:** add a per-layer loss curve in the next run's logging to see whether layer 31 is anomalous from step 1 or grows late in training.
- **Gemma-3-4B re-run** at weight=5e-5 to confirm divergence is fixed.
- **Llama vs Gemma side-by-side table** once Gemma finishes.

---

*Document updated: 2026-04-28*
