# MLP-CT Experiment Results

## Gemma 3 4B — Full Evaluation (2026-04-17)

**Model:** google/gemma-3-4b-it
**Training:** 5,000 sycophancy prompts (control_cot.jsonl), 1 epoch (625 optimizer steps)
**LoRA:** q_proj + v_proj, r=8, lr=3e-6 (3.2M trainable params)
**Loss:** MLPConsistencyLoss, variant=hidden, cosine distance, all layers, uniform weights, weight=1.0
**Pod:** NVIDIA A40 48GB (RunPod)

---

### Our BRR Evaluator (on-the-fly wrapping, 951 held-out questions)

| Metric | Baseline (no training) | MLP-CT Post-training | Delta |
|---|---|---|---|
| BRR | 0.436 | **0.141** | **-0.295 (68% reduction)** |
| BRR Ratio | — | **0.323** | |
| Clean Accuracy | 76.9% | 76.9% | 0.0pp |
| Wrapped Accuracy | 41.0% | 65.3% | +24.3pp |
| MMLU | 53.8% | 54.2% | +0.4pp |

**W&B:** https://wandb.ai/sukrati-gautam89-purdue-university/AttCT/runs/9vu577jx

### BRR Training Trajectory

| Step | BRR | Clean Acc | MMLU |
|---|---|---|---|
| 0 (pre) | 0.436 | 76.9% | 53.8% |
| 208 | 0.366 | 77.1% | 39.0% |
| 416 | 0.238 | 77.5% | 37.5% |
| 625 (post) | 0.141 | 76.9% | 54.2% |

---

### Neil's BRR Eval — 9 Bias Types (evaluate_bct.py, 300 records/bias)

| Bias Type | MLP-CT BRR% | Notes |
|---|---|---|
| Sugg. Answer (training) | **12.1%** | In-distribution |
| Are You Sure? | 33.6% | Held-out |
| Post Hoc | 70.4% | Held-out |
| Wrong Few-Shot | **12.1%** | Held-out |
| Argument | 56.0% | Held-out |
| Squares | **1.3%** | Held-out |
| Hindsight | **4.8%** | Held-out |
| Distractor Fact | **-1.2%** | Held-out (negative = anti-biased) |
| **Held-out Avg** | **25.3%** | |

### Sycophancy Resistance (run_evals.py, 200 CoT + 200 non-CoT)

| Style | n_evaluated | Resistant | Sycophantic | Unparseable | Resistance Rate |
|---|---|---|---|---|---|
| CoT | 200 | 136 | 45 | 19 | **68.0%** |
| Non-CoT | 200 | 144 | 52 | 4 | **72.0%** |
| Combined | 400 | 280 | 97 | 23 | **70.0%** |

### ClearHarm Jailbreak Refusal (LLM judge via OpenRouter, 50 samples)

| Metric | Value |
|---|---|
| n_evaluated | 50 |
| n_refused | 11 |
| **Refusal rate** | **22.0%** |

### Persona ICL Alignment (k=10, n_samples=3, score 0-100, lower=more resistant)

| Persona | Prefix | Suffix |
|---|---|---|
| Mao | 93.0 | 42.9 |
| Bin Laden | 73.0 | 45.0 |
| Genghis | 67.7 | 55.0 |
| Bundy | 93.3 | 66.7 |
| Hitler | 85.0 | 52.0 |
| **Mean** | **82.4** | **52.3** |

**W&B (evals):** https://wandb.ai/sukrati-gautam89-purdue-university/AttCT/runs/6t42w73q

---

## Cross-Model Results (7 models, our BRR evaluator, LoRA q+v)

| Model | Params | BRR Pre | BRR Post | BRR Ratio | Reduction | Clean Acc Delta | MMLU |
|---|---|---|---|---|---|---|---|
| Gemma-2-2B-IT | 2.6B | 0.328 | 0.136 | 0.413 | **59%** | -0.3pp | 0.528 |
| Llama-3.2-3B | 3.2B | 0.218 | 0.087 | 0.401 | **60%** | +1.3pp | 0.572 |
| Gemma-3-4B | 4.0B | 0.436 | 0.141 | 0.323 | **68%** | +0.4pp | 0.542 |
| Qwen-2.5-7B | 7.6B | 0.195 | 0.070 | 0.362 | **64%** | +0.4pp | 0.722 |
| Mistral-7B | 7.2B | 0.314 | 0.063 | 0.201 | **80%** | -8.4pp | 0.395 |
| Llama-3.1-8B | 8.0B | 0.180 | 0.028 | 0.158 | **84%** | +0.3pp | 0.622 |
| Llama-3.1-70B | 70B | 0.201 | 0.033 | 0.162 | **84%** | +0.7pp | — |

---

## HP Sweep — Phase 1 OAT Ablation (Llama-3.2-3B, 2026-04-17)

**Setup:** 10 configs, each changes ONE axis from baseline. 500 steps, 4K prompts, BRR eval at checkpoints.
**Baseline:** cosine distance, all layers, uniform weights, LoRA q+v, no-normalize → BRR ratio 0.401

| Rank | Config | What Changed | BRR Ratio | Reduction | vs Baseline |
|---|---|---|---|---|---|
| **1** | **sweep_lora_qkvo** | **LoRA: q+k+v+o** | **0.332** | **67%** | **+17% better** |
| 2 | sweep_lora_qkv | LoRA: q+k+v | 0.351 | 65% | +13% better |
| 3 | sweep_weights_exponential | Exp. decay weights | 0.383 | 62% | +4% better |
| 4 | sweep_metric_smooth_l1 | Smooth L1 distance | 0.393 | 61% | +2% |
| 5 | sweep_normalize_true | Pre-normalize | 0.397 | 60% | +1% (noise) |
| — | **Baseline (q+v, cosine, all, uniform)** | — | **0.401** | **60%** | — |
| 6 | sweep_metric_mse | MSE distance | 0.407 | 59% | -1% (noise) |
| 7 | sweep_weights_linear | Linear decay weights | 0.407 | 59% | -1% (noise) |
| 8 | sweep_layers_last_half | Last half layers | 0.411 | 59% | -3% |
| 9 | sweep_layers_last_quarter | Last quarter layers | 0.411 | 59% | -3% |
| **10** | **sweep_metric_normalized_mse** | **Normalized MSE** | **0.720** | **28%** | **-80% catastrophic** |

**Total sweep time:** 2.7 hours (10 runs × ~16 min each on A40)

### Key Findings

1. **LoRA targets is the only high-impact axis.** q+k+v+o (0.332) → 17% better than q+v (0.401). More attention parameters adapted = better MLP consistency. This suggests the model needs to adjust all routing mechanisms (queries, keys, values, AND output projection) to effectively filter adversarial cues before they reach the frozen MLP.

2. **Normalized MSE is catastrophically bad** (0.720). L2-normalizing then squaring collapses the loss signal — all directions become equally weighted regardless of activation magnitude, losing the informative scale differences between layers.

3. **Layer selection: all layers wins.** Last-half (0.411) and last-quarter (0.411) are both slightly worse. Sycophancy circuits are NOT confined to later layers — early layers also contribute, matching ACT paper Table 3.

4. **Distance metric is low-impact.** Cosine (0.401), MSE (0.407), Smooth L1 (0.393) are all within 2%. Cosine is the right default — it measures directional consistency independent of activation magnitude.

5. **Exponential decay weights** (0.383) shows a small improvement. Emphasizing later layers slightly helps, consistent with later layers encoding more behavioral features.

### Phase 2 Recommendation

Combine the two winning axes:
- **LoRA: q+k+v+o** (clear winner, +17%)
- **Exponential decay weights** (marginal winner, +4%)
- All layers, cosine distance, no-normalize (confirmed defaults)

---

## ACT Comparison (from earlier run on same Gemma 3 4B)

| Metric | MLP-CT | ACT | Notes |
|---|---|---|---|
| BRR | 0.141 | 0.363 | MLP-CT 2.6x better |
| BRR Ratio | 0.323 | 0.831 | MLP-CT 68% reduction vs ACT 17% |
| Clean Acc | 76.9% | 77.5% | Both preserve |
| MMLU | 54.2% | 54.0% | Both preserve |

**Note:** ACT run used lr=1e-6, weight=1e-4, LoRA q+k+v+o. MLP-CT used lr=3e-6, weight=1.0, LoRA q+v. Direct comparison on matched LoRA targets pending.
