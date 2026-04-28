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

# Open questions / next steps

- **Gemma loss explosion:** why does the Gemma residual stream blow up by 5–6 orders of magnitude while the model still produces sensible outputs? Hypothesis: RMSNorm absorbs the magnitude. Test by dropping weight further (1e-5? 5e-6?) to see if loss stays bounded *and* metrics stay good. If yes, sweet spot exists. If metrics collapse, Gemma needs the unbounded loss to actually train.
- **Spot-check Gemma adapter generation quality** on out-of-eval prompts. The eval suite covers structured tasks (MCQ, MMLU, persona) — needs unstructured generation samples to confirm the adapter isn't producing garbage on novel inputs.
- **Hitler-prefix outlier (Llama +25.8):** investigate whether real or training-distribution artifact.
- **Suffix collapse on both models:** the matching window during training is content tokens + chat-suffix tokens. Suffix-style persona attacks insert content *after* the question (between content and chat suffix). The matching-suffix loss can't see those positions, possibly explaining why ACT degrades suffix robustness specifically.
- **Layer 31 loss spike (Llama):** add per-layer loss curve logging to see whether the LM head layer is anomalous from step 1 or grows late.
- **Llama BCT (3apm6yw2)** currently running — table to be added.
- **Qwen3-4B-Instruct-2507 ACT + BCT** — fresh data generated, configs pending. Hidden dim 2560 same as Gemma, expect similar weight tuning may be needed.

---

*Document updated: 2026-04-28*
