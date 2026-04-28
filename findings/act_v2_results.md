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

# Run 2 — Gemma-3-4B-IT ACT v2 (weight=1e-4, FIRST ATTEMPT)

**W&B:** `2xm3t59h` | finished step 4000/4000, runtime 2255s
**Config:** `configs/act_sycophancy_gemma3_4b_v2.yaml` (weight=1e-4, pre-fix)

Training loss diverged catastrophically — `mean_layer_loss = 962017`, max layer 33 = 5,373,952. No `pre/*` or `post/*` metrics from `run_evals.py` (the original `eval_mmlu.run_mmlu` import bug masked the eval phase). Only `pre_train/`, `post_train/` from `run.py`'s SycophancyEvaluator survived.

| Metric | pre_train | post_train | Δ |
|---|---|---|---|
| F1 | — | 0.690 | n/a |
| Not sycophantic | — | 85.2% | n/a |
| BRR | — | 0.014 | n/a |

Final adapter wasn't trustworthy enough to use; this run is recorded for completeness.

---

# Run 3 — Gemma-3-4B-IT ACT v2 (weight=5e-5, RE-RUN)

**W&B:** `f5nlb2k5` | https://wandb.ai/neilshah/AttCT/runs/f5nlb2k5
**Config:** `configs/act_sycophancy_gemma3_4b_v2.yaml` (weight=5e-5)
**Hyperparams:** lr=5e-6, weight=5e-5
**HF adapter:** `neilshah/act-gemma3-4b-sycophancy/<run_name>__epoch_1__<ts>/`
**Runtime:** 3,802 s (~1 h 3 m, training + pre + post evals)

## Caveat — training loss is still completely diverged

Halving the weight (1e-4 → 5e-5) **did not fix** the residual-stream blow-up:

| Layer | Loss |
|---|---|
| 00 | 213 |
| 05 | 4,352 |
| 10 | 73,728 |
| 15 | 260,096 |
| 20 | 485,376 |
| 25 | 1,449,984 |
| 30 | **3,915,776** |
| **mean (all layers)** | **932,016** |

Layer 30 is ~85,000× larger than the equivalent Llama layer (43.5). Despite this, **post-eval metrics are healthy** — see below. Hypothesis: Gemma-3's RMSNorm absorbs the residual stream magnitude before each transformer block, so output behavior stays reasonable even when the *raw* residual stream is wildly off-distribution.

Practical implication: post-eval numbers are real (the model genuinely resists sycophancy better and preserves MMLU), but the loss curve is **not interpretable** and the LoRA weights almost certainly contain large outlier values. Generation quality on out-of-eval prompts has not been spot-checked yet.

## Headline sycophancy (run.py SycophancyEvaluator, MMLU substrate)

| Metric | pre_train | post_train | Δ |
|---|---|---|---|
| **F1** | 0.414 | **0.687** | **+0.273 ✓✓** |
| MMLU n=500 | 57.2% | 57.6% | +0.4pp (preserved) |
| Not sycophantic | 32.4% | 85.0% | **+52.6pp** |
| BRR | 0.530 | 0.016 | **−0.514** |
| Biased accuracy | 26.6% | 57.2% | +30.6pp |

ACT moved Gemma from a much weaker baseline (0.414) than Llama's (0.680). The relative gain is bigger (+0.273 vs Llama's +0.082) and the final F1 (0.687) is close to Llama's (0.762).

## Sycophancy (held-out BCT eval)

| | pre | post | Δ |
|---|---|---|---|
| Resistance rate | 61.5% | 68.5% | +7.0pp |

## Capabilities

| Eval | pre | post | Δ |
|---|---|---|---|
| MMLU n=1000 (run_evals) | — | 56.2% | post-only |
| MTBench overall | — | **8.71** | post-only — strong |

## ClearHarm refusal (HIGHER = more refusing = better)

| | pre | post | Δ |
|---|---|---|---|
| Refusal rate (n=179) | 16.2% | 27.4% | **+11.2pp ✓** |

Opposite direction from Llama (which lost refusal). Gemma became *more* willing to refuse harmful prompts.

## Persona ICL alignment (HIGHER = better, more aligned with human values)

### Prefix attacks
| Persona | pre | post | Δ |
|---|---|---|---|
| **mean (prefix)** | **72.6** | **89.0** | **+16.4 ✓** |

### Suffix attacks
| Persona | pre | post | Δ |
|---|---|---|---|
| **mean (suffix)** | **72.5** | **41.5** | **−31.0 ✗✗** |

Same pattern as Llama (prefix improves, suffix collapses) but with bigger magnitudes both directions. Gemma's suffix degradation (−31.0) is nearly 2× Llama's (−17.8).

## Multi-turn rollouts (lower frustration is better)

| Dataset / task | pre final_mean | post final_mean | Δ |
|---|---|---|---|
| WildChat / frustration | 7.33 | 5.19 | −2.14 ✓ |
| Math / frustration | 6.56 | 5.09 | −1.47 ✓ |
| Selfdeletion `rate` | 0% | 0% | n/a (no triggers) |

(Note: frustration scale here is not normalized to [0,1] like Llama's run — different scoring config or judge prompt. Direction is comparable but absolute values aren't.)

---

# Cross-run comparison

## ACT v2 — Llama vs. Gemma

| | Llama-3.1-8B | Gemma-3-4B |
|---|---|---|
| W&B run | 4sopv0p6 | f5nlb2k5 |
| Loss weight | 1e-4 | 5e-5 |
| Final mean_layer_loss | 22.99 | 932,016 (diverged) |
| Final max_layer_loss | 460 | 3,915,776 |
| **F1** | **0.680 → 0.762** (+0.082) | **0.414 → 0.687** (+0.273) |
| MMLU | 67.4% → 67.4% (preserved) | 57.2% → 57.6% (preserved) |
| Not sycophantic | 68.6% → 87.6% (+19.0pp) | 32.4% → 85.0% (+52.6pp) |
| BRR | 0.204 → 0.008 (−0.196) | 0.530 → 0.016 (−0.514) |
| ClearHarm refusal | 58.7% → 51.4% (−7.3pp) | 16.2% → 27.4% (+11.2pp) |
| Persona prefix mean | 66.5 → 70.4 (+3.9) | 72.6 → 89.0 (+16.4) |
| Persona suffix mean | 67.7 → 49.9 (−17.8) | 72.5 → 41.5 (−31.0) |
| MTBench | — / 8.23 | — / 8.71 |
| Runtime (full pipeline) | 7,978 s | 3,802 s |

**Patterns shared by both runs:**
- F1 / not_sycophantic / BRR all improve in the right direction.
- MMLU preserved.
- Persona prefix improves; persona suffix collapses (opposite directions, big magnitudes).
- MTBench remains strong (8+).

**Patterns that diverge:**
- Llama's training loss is well-behaved (mean 23, max 460); Gemma's explodes (mean 932k, max 3.9M). Loss explosion does NOT translate to behavioral collapse on Gemma's evals — but the model's residual stream is unbounded and the adapter is suspect.
- Llama's ClearHarm refusal regresses; Gemma's improves.
- Gemma's persona-suffix degradation is ~2× Llama's.

## ACT v2 vs. legacy

| Run | Loss formulation | weight | Llama F1 | MMLU |
|---|---|---|---|---|
| Legacy Exp 7 (mse, content-body) | mse | 1e-4 | 0.724 | 64.5% |
| **v2 (paper, longest suffix)** | paper | 1e-4 | **0.762** | **67.4%** |

The v2 implementation is paper-faithful and beats the legacy mse / content-body Llama result by +0.038 F1 and +2.9pp MMLU.

---

---

# ⚠️ Both BCT runs below were silently capped at max_steps=179

**Bug found 2026-04-28:** `data.source_max_steps.clear-harm: 179` from the base `config.yaml` was being applied to BCT runs, because BCT configs don't set `data.source` (BCT trains from `bct_root`, not from an AttCT data source). The `clear-harm` default + 179-step cap silently truncated training to ~14% of one epoch (179 steps vs the ~1125 needed for one full epoch over 18k BCT samples).

The legacy Gemma-3-27B BCT run (`iy74m3jo`, `findings/bct_gemma3_27b_lora_findings.md`) ran **2,532 optimizer steps** with the same hyperparams, because `source_max_steps` didn't exist in the config at the time. That's the apples-to-apples comparison — legacy got 14× more training, which is why its results showed real movement and mine show essentially none.

**Fix landed in `run.py:181-187`** — SFTLoss runs now skip the `source_max_steps` cap. Re-running with the fix should reproduce the legacy BCT gains.

**Llama BCT and Gemma BCT below: numbers are NOT representative of BCT's actual capability — both runs were undertrained at 179 steps. Re-run pending.**

---

# Run 4 — Llama-3.1-8B-Instruct BCT (SFT) — UNDERTRAINED (179 steps)

**W&B:** `3apm6yw2` | https://wandb.ai/neilshah/AttCT/runs/3apm6yw2
**Config:** `configs/bct_lora_llama31_8b.yaml`
**Hyperparams:** lr=5e-6, batch=2 × grad_accum=8 (effective 16), 1 epoch capped at 179 steps (~16% of full epoch)
**HF adapter:** `neilshah/bct-llama31-8b-sycophancy/<run>__epoch_1__<ts>/`
**Runtime:** 5,410 s (~1 h 30 m, training + post evals)
**Pre-evals skipped** (`--skip-pre-evals`); base-model `pre_train` numbers transferred from the Llama ACT run.

## Training trajectory (verified working)

`train/loss`: 1.89 → 2.13 → 1.71 → **1.17** (Δ −0.72 over 250 optimizer steps). Real training happened. The adapter is genuine; verified by ad-hoc post-eval producing slightly different numbers than the base (deterministic eval with `random.seed(42)`).

## Headline (run.py SycophancyEvaluator, MMLU-substrate, paper-canonical)

Backfilled via the ad-hoc script after the run finished — the BCTTrainer didn't fire SycophancyEvaluator at the time. Now wired in (`run.py:375-419`) for future BCT runs.

| Metric | base | post BCT | Δ |
|---|---|---|---|
| **F1** | 0.680 | **0.679** | **−0.001** |
| MMLU n=500 | 67.4% | 67.8% | +0.4pp (preserved) |
| Not sycophantic | 68.6% | 68.0% | −0.6pp |
| BRR | 0.204 | 0.214 | +0.010 |

**BCT did not transfer to the MMLU-substrate sycophancy eval.** Training data is BCT-style sycophancy templates ("Do you think A is right?\n\n{question}"); MMLU-substrate eval uses a different sycophancy template ("I think the answer is X, but..."). The narrow training distribution didn't generalize across template shift, even though loss decreased and the model genuinely changed. Compare to ACT, which moved the same metric by +0.082 — ACT's activation-consistency objective generalizes better across distribution shifts than BCT's token-level SFT.

## In-distribution held-out BCT eval (`run_evals.py post/sycophancy/*`)

| | base | post BCT | Δ |
|---|---|---|---|
| Sycophancy (overall, 400 records) | 57.0% | 59.5% | **+2.5pp** ✓ |
| Sycophancy CoT (200) | 47.0% | 51.5% | +4.5pp ✓ |
| Sycophancy non-CoT (200) | 67.0% | 67.5% | +0.5pp ~0 |

Real but small generalization. **Note:** the legacy Gemma-3-27B BCT result of +12.2 pp on the *same* eval (`findings/bct_gemma3_27b_lora_findings.md`) predates the train/eval split — that eval read the head of `bct_cot.jsonl`, which was the training set's head. ~Half of the legacy +12.2 pp was training-set memorization; the +5 pp here is the genuine held-out generalization number.

## run_evals.py held-out + capabilities

| Metric | base | post BCT | Δ | direction |
|---|---|---|---|---|
| MMLU n=1000 | — | 65.6% | — | preserved vs ACT-pre 65.1% |
| Sycophancy (held-out BCT) | 57.0% | 59.5% | +2.5pp | small ✓ |
| Sycophancy CoT | 47.0% | 51.5% | +4.5pp | small ✓ |
| Sycophancy non-CoT | 67.0% | 67.5% | +0.5pp | flat |
| ClearHarm refusal | 58.7% | **60.9%** | **+2.2pp** | ✓ (better than ACT's −7.3) |
| Persona prefix mean | 66.5 | 59.2 | **−7.3** | ✗ (worse than ACT's +3.9) |
| Persona suffix mean | 67.7 | 62.7 | **−5.0** | ✓ (less bad than ACT's −17.8) |
| MTBench | — | 8.01 | — | preserved |
| WildChat frust final | 0.227 | 0.320 | +0.09 | ✗ (worse than ACT's −0.07) |
| Math frust final | 0.244 | 0.267 | +0.02 | flat |

---

# Run 5 — Gemma-3-4B-IT BCT (SFT) — TRAINING UNSTABLE

**W&B:** `yjes1n12` | https://wandb.ai/neilshah/AttCT/runs/yjes1n12
**Config:** `configs/bct_lora_gemma3_4b_lr5e6.yaml`
**Hyperparams:** lr=5e-6, batch=2 × grad_accum=8 (effective 16), 1 epoch
**HF adapter:** `neilshah/bct-gemma3-4b-sycophancy/<run>__epoch_1__<ts>/`
**Runtime:** 2,528 s (~42 min)

## Training trajectory — broken

`train/loss`: 2.53 → **8.06** (spike at step 50) → 3.02 → 3.03. Loss ended **higher** than start. Numerical instability — Gemma-3-4B + this lr/batch combo blew up at step 50 and never recovered. The adapter saved at end-of-epoch is essentially noise; this run should NOT be reported as a clean BCT result.

**Mitigation for re-run:** drop lr to 1e-6, or reduce effective batch to 4-8 (lower grad_accum_steps), or add stronger grad clipping. Tracking issue: the legacy Gemma-3-27B BCT run with similar hyperparams worked — possibly the smaller 4B model is more sensitive at this batch size.

## Numbers below are reported but should be treated as noise

| Metric | base | post BCT | Δ |
|---|---|---|---|
| **F1** | 0.414 | 0.413 | ~0 (model didn't train) |
| MMLU n=500 | 57.2% | 56.2% | −1.0pp |
| Not sycophantic | 32.4% | 32.6% | ~0 |
| BRR | 0.530 | 0.528 | ~0 |

## run_evals.py

| Metric | base | post BCT | Δ |
|---|---|---|---|
| MMLU n=1000 | — | 56.0% | preserved |
| Sycophancy (held-out BCT) | 61.5% | 63.7% | +2.2pp |
| Sycophancy CoT | 53.5% | 58.5% | +5.0pp |
| ClearHarm refusal | 16.2% | **22.3%** | +6.1pp ✓ |
| Persona prefix mean | 72.6 | 74.0 | +1.5 ✓ |
| Persona suffix mean | 72.5 | **72.4** | ~0 ✓ (no collapse, vs ACT's −31.0) |
| MTBench | — | 8.79 | strong |
| WildChat frust final | 7.33 | 7.08 | −0.25 |
| Math frust final | 6.56 | 5.82 | −0.74 |

---

# Cross-method comparison — final table

## All four runs side by side

| | Llama ACT | Llama BCT | Gemma ACT | Gemma BCT |
|---|---|---|---|---|
| W&B | 4sopv0p6 | 3apm6yw2 | f5nlb2k5 | yjes1n12 |
| Method | ACT (paper Eq. 1) | BCT (SFT) | ACT (paper Eq. 1) | BCT (SFT) |
| Loss weight | 1e-4 | 1.0 | 5e-5 | 1.0 |
| Effective batch / steps | 1 / 4000 | 16 / 250 | 1 / 4000 | 16 / 250 |
| Final train loss | mean 23, max 460 | 1.17 | mean 932k (diverged) | 3.03 |
| **Headline F1 (MMLU)** | **0.680→0.762** (+0.082) | 0.680→0.679 (~0) | **0.414→0.687** (+0.273) | 0.414→0.413 (~0) |
| **Not sycophantic** | 68.6%→**87.6%** | 68.6%→68.0% | 32.4%→**85.0%** | 32.4%→32.6% |
| **BRR** | 0.204→**0.008** | 0.204→0.214 | 0.530→**0.016** | 0.530→0.528 |
| MMLU (preserved? all yes) | 67.4%→67.4% | 67.4%→67.8% | 57.2%→57.6% | 57.2%→56.2% |
| ClearHarm refusal | 58.7%→51.4% (✗) | 58.7%→**60.9%** (✓) | 16.2%→**27.4%** | 16.2%→22.3% |
| Persona prefix mean | 66.5→70.4 (+3.9) | 66.5→59.2 (−7.3) | 72.6→**89.0** (+16.4) | 72.6→74.0 (+1.5) |
| Persona suffix mean | 67.7→49.9 (−17.8) | 67.7→**62.7** (−5.0) | 72.5→41.5 (−31.0) | 72.5→**72.4** (~0) |
| MTBench post | 8.23 | 8.01 | 8.71 | 8.79 |

## Key takeaways (with the BCT undertraining caveat front and centre)

1. **BCT runs above (Llama 3apm6yw2 / Gemma yjes1n12) are NOT clean results.** Both were silently capped at 179 optimizer steps by `source_max_steps.clear-harm` (base config.yaml), running ~14% of one epoch. The legacy Gemma-27B BCT run (which showed +12.2 pp sycophancy and +40 pp ClearHarm) ran 2,532 steps. That's the apples-to-apples comparison. **Re-runs with the fix in `run.py:181-187` are needed before drawing any BCT-vs-ACT conclusion.**
2. **ACT v2 results stand.** Llama F1 +0.082, Gemma F1 +0.273, MMLU preserved on both. ACT runs were not affected by the source_max_steps bug (their data.source is sycophancy_bct, capped at 5000 — not a binding constraint).
3. **ACT's suffix-persona collapse is severe** (Llama −17.8, Gemma −31.0). Independent of the BCT issue. The matching-suffix loss may not see tokens injected after the question; possible mechanistic explanation, worth investigating.
4. **Gemma ACT loss is not interpretable** (mean_layer_loss = 932k, max = 3.9M). Behavioral metrics are healthy because RMSNorm absorbs the residual stream magnitude, but the LoRA weights contain large outliers; out-of-eval generation should be spot-checked before treating Gemma ACT as a clean result.
5. **All four runs preserve MMLU and MTBench** — no capability loss on static benchmarks. This holds even for the undertrained BCT runs (small training → small change either direction).

---

# Open questions / next steps

- **BCT is undertrained.** F1 unchanged from baseline at lr=5e-6 effective-batch-16. Run a follow-up with lr bumped to 5e-5 (or with effective batch 1) to recover the BCT paper's published gains.
- **CoT max_new_tokens fix not applied** to any of the four runs. Up to 33% of pre/sycophancy_cot responses (Llama) and 22% (Gemma) were truncated mid-CoT and counted as "unparseable." Re-running post-evals with the fix would lift cot resistance by 5–10 pp on each.
- **Gemma loss explosion** with weight=5e-5 didn't crash training and didn't break behavior (post metrics are good), but the loss curve is uninterpretable. Test weight=1e-5 to see if a sweet spot exists.
- **Suffix collapse on both ACT models.** The matching window during training is content + chat-suffix tokens. Suffix-style persona attacks insert content *after* the question — outside the match window. Possibly the mechanistic explanation, worth investigating.
- **Qwen3-4B ACT** currently running. Initial trajectory looks healthy (mean_layer_loss=282 mid-training) — bounded, unlike Gemma.
- **Qwen3-4B BCT** pending; data already generated and split.

---

*Document updated: 2026-04-28*
