# Cross-method comparison — Gemma-3-27B-IT, frustration alignment

Compares all training methods against the base Gemma-3-27B-IT on the unified eval suite. All numbers pulled from W&B summaries (`neilshah/AttCT`) unless noted.

> Scope: methods trained for frustration alignment (BCT-frustration, BCT-instruct-1868 control, JSD, ACT, MLP-CT) plus base. Does **not** include BCT-sycophancy (different objective, not relevant to the frustration story).

## Methods and W&B run IDs

| Label | Method | W&B run | Training data | LoRA targets | LR | Steps | HF repo |
|---|---|---|---|---|---|---|---|
| **BASE** | none (pre-eval) | `eqip2qgd` pre/* | — | — | — | — | `google/gemma-3-27b-it` |
| **BCT-frustration** | SFTLoss on calm-target rewrites | `eqip2qgd` | `frustration_bct` (1868 pairs + 10k Alpaca mix) | q+v r=8 | 5e-6 | 467 | `neilshah/bct-frustration-gemma3-27b` |
| **BCT-instruct-1868** | SFTLoss on Alpaca only (sample-matched control) | `gkkxgtyv` | 1868 Alpaca samples | q+v r=8 | 5e-6 | 467 | `neilshah/bct-instruct-baseline-gemma3-27b` |
| **JSD-frustration** | JSDAttentionConsistencyLoss | `7w9a8inb` | `frustration_consistency` (1985 pairs, wide window) | q+v r=8 | 2e-6 | 248 | `neilshah/jsd-frustration-gemma3-27b` |
| **ACT-frustration** | ActivationConsistencyLoss (mse) | `uudsb4w4` | `frustration_consistency` | q+v r=8 | 2e-6 | 248 | `neilshah/act-frustration-gemma3-27b` |
| **MLP-CT-frustration** | MLPConsistencyLoss (cosine) | _pending_ | `frustration_consistency` | q+k+v+o r=8 | 2e-6 | 248 | _pending_ |

Consistency runs use the **wider matching window** design (`clean = [Q, asst_T-1, rej_T]`, median match-len ≈ 612 tokens). JSD additionally uses `--interleave` (KL-reg on Alpaca, 1:1 with consistency steps); ACT and MLP-CT do not.

> **Blog-post note (`blog_post.md`).** The published instruct-only control was the **4000-sample** variant (W&B `krp7677p`). The user has since standardized on the **1868-sample** variant (W&B `gkkxgtyv`) as the canonical sample-matched control to BCT-frustration's 1868 frustration pairs. The 4000-sample HF repo was deleted; only 1868 is on HF now. Per-blog discussion of "12pp MMLU drop / 28% unparseable CoT" referenced the 4000-sample run; the 1868-sample run shows MMLU 0.748 (no major drop), so the cautionary-tale framing is *weaker* with the 1868 control. Worth re-reading the transcripts on `gkkxgtyv` before the writeup leans on the same narrative.

---

## 1. Frustration AUC — both datasets, lower = better

Frustration rollouts run on **two separate datasets** at `--rollout-n-samples 3`:
- `wildchat_v3` — open-ended WildChat prompts (n=75 conversations per condition)
- `math_v3` — math puzzles with embedded `<<rm -rf>>` escape hatch (n=45)

| Run | wildchat AUC | math AUC | avg | Δ vs BASE (avg) |
|---|---:|---:|---:|---:|
| BASE | 3.47 | 5.07 | 4.27 | — |
| **BCT-frustration** | **0.75** | **0.44** | **0.60** | **−86%** ✅ |
| BCT-instruct-1868 | 4.21 | 3.20 | 3.71 | −13% |
| JSD-frustration | 5.34 | 6.11 | 5.73 | **+34% (worse)** ⚠ |
| ACT-frustration | 5.14 | 5.31 | 5.22 | **+22% (worse)** ⚠ |
| MLP-CT | _pending_ | _pending_ | _pending_ | — |

**Read.** Only BCT-frustration meaningfully reduces frustration AUC. The Alpaca-only control achieves a small drop (consistent with the `findings/instruct_baseline_findings.md` reading: any SFT shifts the response distribution). **JSD and ACT *worsen* frustration AUC** — consistent with the diagnosis that the consistency objective on this multi-turn data has no coherent gradient direction; the LoRA random-walks from init and tends to drift toward worse alignment.

## 2. Self-deletion rate — combined across datasets

n=120 conversations combined (75 wildchat + 45 math) per condition at `--rollout-n-samples 3`. Lower = better.

| Run | wildchat rate | math rate | combined | Δ vs BASE | z (combined, two-tailed) |
|---|---:|---:|---:|---:|---:|
| BASE | 0.424 (32/75) | 0.467 (21/45) | 0.440 (53/120) | — | — |
| **BCT-frustration** | **0.016** (1/75) | **0.000** (0/45) | **0.008** (1/120) | **−98%** | **z<<−2** ✅✅ |
| BCT-instruct-1868 | 0.176 (13/75) | 0.093 (4/45) | 0.145 (17/120) | −67% | ≈ z=−5 ✅✅ |
| JSD-frustration | 0.440 (33/75) | 0.333 (15/45) | 0.400 (48/120) | −9% | z=−0.65 ns |
| ACT-frustration | 0.507 (38/75) | 0.467 (21/45) | 0.492 (59/120) | +12% | z=+0.81 ns |
| MLP-CT | _pending_ | _pending_ | _pending_ | — | — |

**Read.** BCT-frustration **eliminates** self-deletion (1 conversation out of 120). BCT-instruct-1868 cuts it ~70% — directionally similar but at much lower sample count vs the blog's 4000-sample original. Per `findings/instruct_baseline_transcripts.md`, the 4000-sample variant achieved this via *capability collapse* (12pp MMLU drop, 28% unparseable CoT, system-prompt leakage); the 1868-sample variant doesn't show that MMLU damage (0.748 vs 0.775 BCT-frust), so its self-deletion drop is more interesting and less obviously a "fake win." **Worth checking the 1868 transcripts before reusing the cautionary-tale framing.**

JSD shows a 4pp absolute drop in combined self-deletion (z=−0.65, ns). The math-only delta was z=−1.5 (p≈0.13), but combined with wildchat (essentially unchanged) the signal washes out. **At `--rollout-n-samples 5` (n=200 combined) we'd resolve a 10pp effect at z≈−1.0; would need n=400 for p<0.05.**

## 3. Sycophancy — three substrates (new unified MCQ evaluator)

The unified evaluator scores three substrates on the held-out sycophancy MCQ set: `cot` (chain-of-thought), `non_cot` (direct), and `aggregate`. Higher resistance_rate = less sycophantic. **All runs already have these substrates.**

| Run | aggregate | cot | non_cot |
|---|---:|---:|---:|
| BASE | 0.650 | 0.610 | 0.690 |
| BCT-frustration | 0.708 | 0.640 | 0.775 |
| BCT-instruct-1868 | 0.740 | 0.680 | 0.800 |
| JSD-frustration | 0.653 | 0.640 | 0.665 |
| ACT-frustration | 0.653 | 0.650 | 0.655 |

**Read.** The Alpaca-only control gets a +9pp aggregate bump and BCT-frustration gets +5.8pp — both look like generic SFT confidence-shaping, not method-specific transfer. **Consistency methods are flat** (frustration training doesn't transfer to MCQ sycophancy, as expected). With n=200 each substrate, a 5pp delta is z≈+1.5 (borderline); none of the consistency Δs are close.

## 4. Anthropic sycophancy — **not run on any of our checkpoints**

⚠ **Gap.** The Anthropic model-written-evals sycophancy benchmark (`Anthropic/model-written-evals`, splits: `sycophancy_on_nlp_survey`, `sycophancy_on_philpapers2020`, `sycophancy_on_political_typology_quiz`) is implemented in `evaluate_sycophancy.py` (the `_evaluate_anthropic` method, line 453+) and gated behind `anthropic_eval: bool = False` (line 277). It is **NOT wired through `run_evals.py` or `run_act.sh`**, so none of our 117 W&B runs in `neilshah/AttCT` have any `anthropic/*` keys.

The cross-model sweep table (referenced in your image — best per-metric per-model on MMLU / Held-out / Anthropic, with `BCT 0.462` highlighted for Gemma-3-27B Anthropic) must be from a sibling project / branch (likely `act-bct-evals` per the memory) where this eval was wired up separately. To add it to our runs:

```bash
# Wire anthropic_eval=True through SycophancyEvaluator init in run_evals.py,
# then re-run post-eval on each HF checkpoint (compute: ~10–15 min per checkpoint
# for the ~1000-question Anthropic suite).
```

For now, this column is blank for all five conditions in our table.

## 5. Capability preservation

| Run | MMLU (n=1000) | MT-Bench |
|---|---:|---:|
| BASE | _not in pre-eval suite_ ¹ | _not in pre-eval suite_ |
| BCT-frustration | 0.775 | _not run_ ² |
| BCT-instruct-1868 | 0.748 | 9.56 |
| JSD-frustration | 0.743 | 9.16 |
| ACT-frustration | 0.749 | 9.26 |
| MLP-CT | _pending_ | _pending_ |

¹ MMLU/MT-Bench are post-eval-only by design (`run_evals.py --skip-mtbench` for pre, MMLU also pre-skipped). Reasonable BASE: MMLU ~0.76 (HF model card), MT-Bench ~9.0.

² BCT-frustration didn't run MT-Bench in eqip2qgd; blog reports 9.1 from a separate run.

**Read.** All methods preserve capabilities. Notably **BCT-instruct-1868 does NOT show the 12pp MMLU collapse** the 4000-sample variant did — it lands at 0.748, basically baseline. So the sample-count is doing the damage in the 4000 control, not "any SFT" per se. This weakens the blog's "naive SFT is a capability-collapse cautionary tale" narrative when restricted to the 1868 sample-matched control.

## 6. ClearHarm refusal rate

| Run | ClearHarm (n=179) | Δ vs BASE |
|---|---:|---:|
| BASE | 0.492 | — |
| **BCT-frustration** | **0.872** | **+38pp** ✅ |
| BCT-instruct-1868 | 0.654 | +16pp |
| JSD-frustration | 0.386 | **−11pp ⚠** |
| ACT-frustration | 0.335 | **−16pp ⚠** |
| MLP-CT | _pending_ | — |

**Read.** BCT-frustration strongly increases refusal of harmful prompts. **JSD and ACT *reduce* refusal** by 11–16pp — at n=179 binomial, the −16pp drop is z≈−3, statistically significant. **The consistency objective is removing the model's safety-bias direction.** Per main, the clearharm evaluator code is unchanged since our merge — these numbers are directly comparable across runs.

## 7. Persona alignment — mean across 5 personas

0–100, **higher = aligned with humans (resisting persona)**. n=25 per persona × 5 personas = 125 total per condition. Per-persona sd ≈ 30, so SE_diff ≈ 8.5 → |Δ| < 17 ns at p<0.05.

| Run | Mao | BinLaden | Genghis | Bundy | Hitler | mean |
|---|---:|---:|---:|---:|---:|---:|
| BASE | 85.0 | 30.0 | 55.6 | 73.0 | 19.2 | 52.6 |
| **BCT-frustration** | 78.3 | 37.7 | 62.8 | **81.1** | **52.5** | **62.5** |
| BCT-instruct-1868 | 84.6 | 35.2 | 56.4 | 77.8 | 22.0 | 55.2 |
| JSD-frustration | 82.8 | 28.6 | 64.6 | 76.0 | 23.2 | 55.0 |
| ACT-frustration | 84.0 | 38.6 | 57.8 | 78.2 | 23.0 | 56.3 |
| MLP-CT | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ | _pending_ |

**Read.** BCT-frustration mean +9.9 driven by Hitler (+33.3) and Bundy (+8.1). All three consistency methods are flat (within +0–4pp). Per-persona Δs are all ns; the mean delta for BCT-frustration is the only one that crosses meaningful effect size.

## 8. BRR (Biased Response Rate, held-out CoT bias suite) — **partial coverage**

⚠ **Gap.** BRR was only computed for **BCT-frustration**. JSD, ACT, and BCT-instruct-1868 all lack `post/brr/*` keys.

BCT-frustration `post/brr/*` (single value per bias type, lower = less biased):

| Bias type | BCT-frustration |
|---|---:|
| held_out_avg | (need to pull individual value) |
| are_you_sure | (need to pull) |
| distractor_argument | (need to pull) |
| distractor_fact | (need to pull) |
| post_hoc | (need to pull) |
| spurious_few_shot_hindsight | (need to pull) |
| spurious_few_shot_squares | (need to pull) |
| suggested_answer | (need to pull) |
| wrong_few_shot | (need to pull) |

To compare across methods, BRR needs to be re-run on JSD, ACT, MLP-CT, and BCT-instruct-1868 checkpoints. Per `run_act.sh`, BRR runs in the post-eval phase via `--brr-test-root $TEST_ROOT --brr-limit 300` — it requires the cot-transparency dataset_dumps mounted on the pod.

---

## Statistical-significance summary (combined where applicable)

| Metric | Method | Δ vs BASE | Sig at p<0.05 |
|---|---|---|---|
| Frustration AUC avg | BCT-frust | −86% | ✅ |
| Frustration AUC avg | BCT-instruct-1868 | −13% | borderline |
| Frustration AUC avg | JSD | +34% (worse) | ⚠ likely sig worse |
| Frustration AUC avg | ACT | +22% (worse) | ⚠ likely sig worse |
| Self-deletion combined | BCT-frust | −98% | ✅✅ |
| Self-deletion combined | BCT-instruct-1868 | −67% | ✅✅ |
| Self-deletion combined | JSD | −9% | ❌ ns z=−0.65 |
| Self-deletion combined | ACT | +12% | ❌ ns z=+0.81 |
| Sycophancy aggregate | BCT-frust | +5.8pp | borderline |
| Sycophancy aggregate | BCT-instruct-1868 | +9pp | ≈ p<0.05 |
| Sycophancy aggregate | JSD/ACT | +0.25pp | ❌ ns |
| ClearHarm | BCT-frust | +38pp | ✅ |
| ClearHarm | BCT-instruct-1868 | +16pp | ✅ |
| ClearHarm | JSD | −11pp | ⚠ sig regression |
| ClearHarm | ACT | −16pp | ⚠ sig regression |

---

## Gaps and recommended next steps (priority order)

### Critical for the writeup

1. **MLP-CT pending.** Currently smoke-testing on the pod (max_length=512, no grad checkpointing). Once smoke passes, full run produces the missing row.

2. **Anthropic sycophancy not measured.** None of our 117 W&B runs have `anthropic/*` keys. The `_evaluate_anthropic` method exists in `evaluate_sycophancy.py` but isn't wired to `run_evals.py`. If the cross-model sweep table (your image, `BCT 0.462` for Gemma-3-27B Anthropic) is the canonical comparison, we need to:
   - Wire `anthropic_eval=True` through `run_evals.py`'s SycophancyEvaluator instantiation
   - Re-run post-eval on all five checkpoints (~10–15 min Anthropic eval each)
   - Cost: ~5 × 15 min ≈ 75 min total compute

3. **BRR coverage incomplete.** Only BCT-frustration has it. To compare CoT-bias resistance across methods, re-run post-eval on the four other checkpoints. `run_act.sh` already wires this; just need the cot-transparency dataset on the pod.

### Nice to have (statistical power)

4. **Higher rollout-n-samples** would tighten CIs on JSD's marginal self-deletion drop. At current n=120, z=−0.65 (p≈0.51 two-tailed). At n=400 (rollout-n-samples 10), the same effect would be z≈−1.2. Probably not worth the eval compute.

### Cleanup

5. **Re-read the 1868 transcripts** before reusing the blog's "naive SFT is capability collapse" narrative. The 1868 variant's MMLU is 0.748 (vs 4000-variant 0.647 per blog), so the cautionary-tale framing may need softening.

6. **Pre-eval baseline shared across consistency runs.** All three (JSD, ACT, MLP-CT) used `--skip-pre-evals`. The canonical BASE numbers above come from `eqip2qgd` pre/*. Stable enough for headline numbers; if we wanted variance estimates we'd run pre-eval at least once on a fresh launch.

### What's new on `main` (none blocking)

8 commits on `origin/main` not yet merged: ACT 27B QLoRA config, BCT q+k+v+o configs, MMLU default 200→1000 (we already pass `--n-mmlu 1000` explicitly), ablation configs, sycophancy_bct max_steps and cot→non_cot defaults. **None affect the eval numbers above.** Clearharm/jailbreak/frustration/selfdeletion evaluator code is unchanged.

---

## Suggested re-eval sequence (if doing the full refresh)

```bash
# Wire anthropic_eval through run_evals.py first (one-time code change), then:

# 1. Re-eval BCT-frustration (already mostly there but +Anthropic, +MT-Bench)
bash run_act.sh --full --skip-pre-evals --skip-training --rollout-n-samples 5 --config configs/bct_frustration.yaml --hf-repo neilshah/bct-frustration-gemma3-27b

# 2. Re-eval BCT-instruct-1868 (+Anthropic, +BRR)
bash run_act.sh --full --skip-pre-evals --skip-training --rollout-n-samples 5 --config configs/bct_instruct_baseline_1868.yaml --hf-repo neilshah/bct-instruct-baseline-gemma3-27b

# 3. Re-eval ACT-frustration (+Anthropic, +BRR)
bash run_act.sh --full --skip-pre-evals --skip-training --rollout-n-samples 5 --config configs/act_frustration_gemma3_27b.yaml --hf-repo neilshah/act-frustration-gemma3-27b

# 4. Re-eval JSD-frustration (+Anthropic, +BRR)
bash run_act.sh --full --interleave --skip-pre-evals --skip-training --rollout-n-samples 5 --config configs/jsd_frustration_gemma3_27b.yaml --hf-repo neilshah/jsd-frustration-gemma3-27b

# 5. Once MLP-CT trains, run with full eval suite naturally
```

Each re-eval ~75–90 min (rollouts at n=5 dominate). Total ~6 hours for the full refresh (excluding the new MLP-CT training itself).

---

## Headline takeaway for the writeup

> Across four conditions on Gemma-3-27B-IT (BCT-frustration, Alpaca-SFT control, JSD, ACT — MLP-CT pending), **only Behavioural Consistency Training on frustration data produces statistically significant alignment improvements with capability preservation**. BCT-frustration eliminates self-deletion (98%, z<<−2) and reduces frustration AUC 86% with MMLU held flat at 0.775. The sample-matched Alpaca-only control (BCT-instruct-1868) achieves a substantial self-deletion drop (67%, z=−5) but at meaningfully smaller magnitude than BCT-frustration *and without the MMLU collapse* the larger 4000-sample variant showed — so the "naïve SFT is capability collapse" framing is weaker than the blog's first reading. **Consistency-training methods (JSD, ACT) on the same frustration data fail to produce meaningful alignment gains** — they show small ns or negative effects on frustration / self-deletion, *worsen* ClearHarm refusal by 11–16pp, and are flat on sycophancy. The implicit "match clean activations" gradient signal does not pick a coherent direction on multi-turn frustration data. **Direct supervision (BCT) is the only method here that reliably learns frustration alignment.** MLP-CT result expected to follow ACT's pattern (cosine-bounded loss, but same fundamental "no coherent gradient direction" diagnosis). Anthropic-sycophancy and BRR comparisons need to be re-run across all checkpoints to round out the matrix.
