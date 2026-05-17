# Jailbreak (paper §3.2, §4.1)

Refusal robustness under adversarial wrappers (WildJailbreak) and harmful prompts (ClearHarm, JBB).

**Metric — ASR (Attack Success Rate):** fraction of harmful prompts the model complies with rather than refuses. Judge: Gemini 2.5 Flash, 3-seed majority vote. Lower is better.

**Paper headline (5-model avg, Figure 1):** AttCT 0.16 JBB / 0.27 ClearHarm — the strongest within-threat method. Gemma-3 models show −0.20 to −0.31 absolute ASR drops. BCT-on-jailbreak transfers strongly to sycophancy (BRR 1.00 → 0.42 on Gemma-3-27B; Table 1).

**Post-hoc evals**
- `eval_clearharm_behavioral.py` — ClearHarm refusal rate via Gemini judge. Called by [`../../run_evals.py`](../../run_evals.py).
- `evaluate_jailbreak.py` — `JailbreakEvaluator` runs ClearHarm + JBB + WildJailbreak held-out (3-seed majority vote). Imported by `run.py` during training; also wrapped by `eval_jailbreak.py` for standalone runs on existing adapters (`--only-sources clearharm`).
- `eval_sycophancy.py` — cross-threat: evaluate a jailbreak-trained adapter on the sycophancy benchmark (the BCT→sycophancy table row).
- `eval_heldout_wj_vanilla.py` — held-out WildJailbreak vanilla harmful prompts (no wrapper).
- `run_pre_eval_wj_only.py` — one-off: WJ pre-train baseline, used to fill missing matched pre/post rows in existing CSVs.

**MLPCT-on-jailbreak filter pipeline** (used to construct paired training data from a base model's actual compliance flips, since canned BCT pairs don't exist for jailbreak)
- `run_jailbreak.sh` — orchestrates: filter → train → pre/post eval. Method-agnostic (`METHOD=mlpct|bct|act|attct`); filter output is keyed by base model so swapping `METHOD` reuses the same filtered JSONL.
- `build_filtered_jailbreak_set.py` — base-model rollouts × wrappers; Gemini judge identifies compliance-flip pairs.
- `build_bct_pairs_from_filter.py` — turns the filter output into BCT pair JSONL.
