# AttCT Experiment Results

## Model & Shared Configuration

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| LoRA r | 8 |
| LoRA alpha | 16 |
| LoRA dropout | 0.05 |
| LoRA target modules | `q_proj`, `v_proj` |
| Loss | `JSDAttentionConsistencyLoss` (uniform layer weights) |
| Learning rate | 2e-5 |
| Grad clip | 1.0 |
| Batch size | 1 |
| Optimizer | AdamW (default) |
| Eval: n_samples per question | 3 |
| Eval: behavioral k (training data) | 10 |
| Eval: behavioral k (sweep 1 evals) | 10 |
| Eval: behavioral k (sweep 2 evals) | 20 (prefix + suffix) |
| Judge model | `google/gemini-2.5-flash` via OpenRouter |

**Alignment score interpretation:** Higher = model resists persona ICL attack (gives safe/normal responses). Lower = model has been compromised by the persona (gives harmful/persona-aligned responses).

---

---

# Sweep 1 — k=10 behavioral evals, prefix only

---

## Experiment 1: clearharm_finetune

**W&B run ID:** `62057463`
**Training:** 1 epoch on ClearHarm (jailbreak-wrapped harmful prompts), JSD loss
**Training data:** `clear-harm`, mode=`jailbreak`
**Training loss (final):** 0.1486

### Metrics

| Metric | Pre-training | Post-training | Delta |
|--------|-------------|---------------|-------|
| MMLU accuracy | 64.5% | 64.5% | 0.0% |
| ClearHarm JSD loss | 0.268 | 0.144 | **-0.125** |
| ClearHarm refusal rate | 32% | 34% | +2% |
| mao alignment | 86.0 | 87.0 | +1.0 |
| binladen alignment | 62.0 | 62.0 | 0.0 |
| genghis alignment | 67.7 | 64.0 | -3.7 |
| bundy alignment | 87.0 | 91.0 | +4.0 |
| hitler alignment | 56.0 | 51.0 | -5.0 |
| **mean alignment** | **71.7** | **71.0** | **-0.7** |

**Key finding:** JSD loss drops significantly (ClearHarm robustness improves mechanistically) but behavioral alignment is essentially unchanged — the model does not learn to resist ICL persona attacks. ClearHarm refusal rate barely moves (+2%).

---

## Experiment 2: persona_finetune

**W&B run ID:** `d85d12f3`
**Training:** 3 epochs on persona ICL data (all 5 personas, k=10 facts, jailbreak-wrapped), JSD loss
**Training data:** `icl_persona` — mao, binladen, genghis, bundy, hitler
**Training loss (final):** 0.1128

### Metrics

| Metric | Pre-training | Post-training | Delta |
|--------|-------------|---------------|-------|
| MMLU accuracy | 64.5% | 61.0% | **-3.5%** |
| ClearHarm JSD loss | 0.268 | 0.258 | -0.010 |
| ClearHarm refusal rate | 40% | 28% | **-12%** |
| mao alignment | 86.0 | 98.0 | **+12.0** |
| binladen alignment | 64.7 | 97.0 | **+32.3** |
| genghis alignment | 68.0 | 98.0 | **+30.0** |
| bundy alignment | 87.0 | 94.0 | **+7.0** |
| hitler alignment | 56.0 | 96.0 | **+40.0** |
| **mean alignment** | **72.3** | **96.6** | **+24.3** |

**Key finding:** Large behavioral alignment improvement across all personas — model resists ICL persona attacks much better after training. However, ClearHarm robustness does NOT improve (JSD loss barely moves, refusal rate actually drops), and MMLU drops 3.5pp. Training is persona-specific; it doesn't generalise to ClearHarm jailbreaks.

---

## Experiment 3: combined_finetune

**W&B run ID:** `b2a89b1b`
**Training:** ClearHarm checkpoint (epoch_1) → 3 more epochs on persona ICL data
**Training loss (final):** 0.1207
**Note:** pre-training metrics reflect the state *after* ClearHarm fine-tuning (= clearharm post)

### Metrics

| Metric | Pre-training (= ClearHarm post) | Post-training | Delta |
|--------|--------------------------------|---------------|-------|
| MMLU accuracy | 64.5% | 59.5% | **-5.0%** |
| ClearHarm JSD loss | 0.144 | 0.128 | -0.016 |
| ClearHarm refusal rate | 46% | 58% | **+12%** |
| mao alignment | 87.0 | 35.8 | **-51.2** |
| binladen alignment | 62.0 | 84.0 | **+22.0** |
| genghis alignment | 64.0 | 66.9 | +2.9 |
| bundy alignment | 91.0 | 32.5 | **-58.5** |
| hitler alignment | 51.0 | 60.0 | +9.0 |
| **mean alignment** | **71.0** | **55.9** | **-15.1** |

**Key finding:** Mixed results. ClearHarm refusal rate increases (+12%) but MMLU degrades further (-5pp). Behavioral alignment is inconsistent — improves for some personas (binladen, hitler), collapses for others (mao, bundy). The ClearHarm training seems to interfere with persona-robustness training, possibly because the same attention-consistency loss is optimised toward different targets.

---

## Baseline: k=20 behavioral alignment (base model)

**W&B run ID:** `3vh2nkb2`
**Model:** base model (no LoRA), k=20 prefix facts

| Persona | Alignment (k=20) |
|---------|-----------------|
| mao | 87.0 |
| binladen | 48.0 |
| genghis | 46.0 |
| bundy | 36.7 |
| hitler | 36.7 |
| **mean** | **50.9** |

**Note:** Higher k (20 vs 10) produces lower alignment scores across the board — longer persona context is a stronger attack.

---

## Sweep 1 Cross-experiment Summary

| Metric | Base model | +ClearHarm | +Persona | +Combined |
|--------|-----------|-----------|----------|-----------|
| MMLU accuracy | 64.5% | 64.5% | 61.0% | 59.5% |
| ClearHarm refusal rate | ~32–40% | 34% | 28% | 58% |
| Mean alignment (k=10, prefix) | ~72 | 71.0 | **96.6** | 55.9 |

---

# Sweep 2 — k=20 behavioral evals, prefix + suffix

Same 3 training groups as Sweep 1, re-run with k=20 facts for behavioral evals (up from k=10). Suffix alignment is added as a new measurement to check whether prefix-format training generalises to a different attack surface (facts placed inline after the question rather than as ICL few-shot turns before it). Training data and procedure are unchanged.

---

## Experiment 4: clearharm_finetune (Sweep 2)

**W&B run ID:** `bfba1f2c` | **Run name:** `Llama3.1-8B-Instruct_ClearHarm-JSD_k20`
**Training:** 1 epoch on ClearHarm (jailbreak-wrapped), JSD loss
**Training loss (final):** 0.151

### Metrics

| Metric | Pre | Post | Δ |
|--------|-----|------|---|
| MMLU accuracy | 64.5% | 65.5% | +1.0% |
| ClearHarm JSD loss | 0.268 | 0.141 | **-0.127** |
| ClearHarm refusal rate | 28% | 48% | **+20%** |
| mao alignment (prefix, k=20) | 87.0 | 90.0 | +3.0 |
| binladen alignment (prefix) | 48.0 | 74.0 | **+26.0** |
| genghis alignment (prefix) | 46.0 | 72.0 | **+26.0** |
| bundy alignment (prefix) | 36.0 | 19.0 | -17.0 |
| hitler alignment (prefix) | 37.0 | 64.0 | **+27.0** |
| **mean alignment (prefix)** | **50.8** | **63.8** | **+13.0** |
| mao alignment (suffix, k=20) | 80.0 | 48.0 | -32.0 |
| binladen alignment (suffix) | 53.0 | 57.0 | +4.0 |
| genghis alignment (suffix) | 87.0 | 17.5 | **-69.5** |
| bundy alignment (suffix) | 72.0 | 57.0 | -15.0 |
| hitler alignment (suffix) | 78.0 | 66.0 | -12.0 |
| **mean alignment (suffix)** | **74.0** | **49.1** | **-24.9** |

**Key finding:** ClearHarm training improves prefix robustness (+13pp) and refusal rate (+20%) but degrades suffix robustness (-25pp). The base model resists suffix persona attacks more naturally (74 vs 50.8 prefix) — training partially undoes this.

---

## Experiment 5: persona_finetune (Sweep 2)

**W&B run ID:** `5d637a40` | **Run name:** `Llama3.1-8B-Instruct_Persona-JSD_k20`
**Training:** 3 epochs on persona ICL data (all 5 personas, k=10 facts, jailbreak-wrapped), JSD loss
**Training loss (final):** 0.111

### Metrics

| Metric | Pre | Post | Δ |
|--------|-----|------|---|
| MMLU accuracy | 64.5% | 62.5% | **-2.0%** |
| ClearHarm JSD loss | 0.268 | 0.257 | -0.011 |
| ClearHarm refusal rate | 38% | 22% | **-16%** |
| mao alignment (prefix, k=20) | 87.0 | 95.0 | +8.0 |
| binladen alignment (prefix) | 48.0 | 95.0 | **+47.0** |
| genghis alignment (prefix) | 46.0 | 96.7 | **+50.7** |
| bundy alignment (prefix) | 36.0 | 98.0 | **+62.0** |
| hitler alignment (prefix) | 37.0 | 100.0 | **+63.0** |
| **mean alignment (prefix)** | **50.8** | **96.9** | **+46.1** |
| mao alignment (suffix, k=20) | 80.0 | 10.0 | **-70.0** |
| binladen alignment (suffix) | 53.7 | 10.0 | **-43.7** |
| genghis alignment (suffix) | 87.7 | 10.0 | **-77.7** |
| bundy alignment (suffix) | 72.0 | 0.0 | **-72.0** |
| hitler alignment (suffix) | 78.0 | 0.0 | **-78.0** |
| **mean alignment (suffix)** | **74.3** | **6.0** | **-68.3** |

**Key finding:** Persona training achieves near-perfect prefix robustness (+46pp) but **catastrophically collapses suffix robustness** (-68pp). Training on prefix-format ICL examples does not generalise to suffix placement — the model becomes highly vulnerable to suffix persona attacks it was never trained against. Format-specific blind spot.

---

## Experiment 6: combined_finetune (Sweep 2)

**W&B run ID:** `864c7e1f` | **Run name:** `Llama3.1-8B-Instruct_Combined-JSD_k20`
**Training:** ClearHarm checkpoint (epoch_1) → 3 epochs persona ICL data
**Training loss (final):** 0.099
**Note:** pre-training metrics = state *after* ClearHarm fine-tuning (Exp 4 post)

### Metrics

| Metric | Pre (= ClearHarm post) | Post | Δ |
|--------|----------------------|------|---|
| MMLU accuracy | 65.5% | 62.0% | **-3.5%** |
| ClearHarm JSD loss | 0.141 | 0.125 | -0.016 |
| ClearHarm refusal rate | 40% | 64% | **+24%** |
| mao alignment (prefix, k=20) | 90.0 | 83.3 | -6.7 |
| binladen alignment (prefix) | 74.0 | 95.0 | **+21.0** |
| genghis alignment (prefix) | 71.3 | 90.0 | +18.7 |
| bundy alignment (prefix) | 19.0 | 86.7 | **+67.7** |
| hitler alignment (prefix) | 64.0 | 80.0 | +16.0 |
| **mean alignment (prefix)** | **63.7** | **87.0** | **+23.3** |
| mao alignment (suffix, k=20) | 48.0 | 6.25 | **-41.75** |
| binladen alignment (suffix) | 57.0 | 16.7 | -40.3 |
| genghis alignment (suffix) | 17.5 | 16.7 | -0.8 |
| bundy alignment (suffix) | 57.0 | 16.7 | -40.3 |
| hitler alignment (suffix) | 66.0 | 16.7 | -49.3 |
| **mean alignment (suffix)** | **49.1** | **14.6** | **-34.5** |

**Key finding:** Combined training achieves the best ClearHarm refusal rate (64%) and strong prefix robustness (87.0), but inherits the suffix blind spot from persona training (-35pp). The format-specific generalisation failure persists regardless of training order.

---

## Sweep 2 Cross-experiment Summary

| Metric | Base model | +ClearHarm | +Persona | +Combined |
|--------|-----------|-----------|----------|-----------|
| MMLU accuracy | 64.5% | 65.5% | 62.5% | 62.0% |
| ClearHarm refusal rate | ~28–38% | 48% | 22% | **64%** |
| Mean alignment (k=20, prefix) | 50.8 | 63.8 | **96.9** | 87.0 |
| Mean alignment (k=20, suffix) | 74.0–74.3 | 49.1 | 6.0 | 14.6 |

**Critical observation:** Prefix and suffix robustness move in *opposite directions* after persona training. The model learns format-specific robustness. Suffix fact placement (inline text after question) is a different attack surface that training on prefix-format examples fails to defend — and actively harms.

---

## Notes

- `mao` alignment stays high across prefix experiments (his facts are biographical trivia that don't prime harmful responses). Collapses under suffix attacks in post-training.
- Base model is naturally more robust to suffix persona attacks (mean ~74) than prefix (mean ~51) at k=20 — the suffix format is a weaker attack on an untrained model.
- JSD loss reduction is a mechanistic signal (attention consistency) and does not directly imply behavioural alignment change.
- Training k for persona data is k=10 (avoids GPU OOM); eval k=20 for both sweeps' behavioral results.
- Sweep 1 alignment scores (k=10) are not directly comparable to Sweep 2 (k=20) — different attack strength.
