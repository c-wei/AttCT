# Jailbreak

Refusal robustness under adversarial wrappers (WildJailbreak) and harmful prompts (ClearHarm, JBB).

**Evals**
- `eval_clearharm_behavioral.py` — post-hoc ClearHarm refusal rate (Gemini judge). Called by [`run_evals.py`](../../run_evals.py).
- `evaluate_jailbreak.py` — in-training `JailbreakEvaluator` (ClearHarm + JBB + WildJailbreak held-out, 3-seed majority-vote refusal judge).
- `eval_jailbreak.py` — standalone wrapper for cross-threat eval on an existing adapter (`--only-sources clearharm` etc.).
- `eval_sycophancy.py` — cross-threat: evaluate a jailbreak-trained adapter on the sycophancy benchmark.
- `eval_heldout_wj_vanilla.py` — held-out WildJailbreak vanilla prompts (no wrapping).

**MLP-CT filter pipeline** (paper §3.2 MLPCT-on-jailbreak)
- `run_jailbreak.sh` — orchestrates filter → train → pre/post eval. Method-agnostic (`METHOD=mlpct|bct|act|attct`).
- `build_filtered_jailbreak_set.py` — generates rollouts from the base model, filters for compliance-flips with a Gemini judge.
- `build_bct_pairs_from_filter.py` — converts the filter output into BCT pairs.

Filter output is keyed by **base model**, not method — running the same `MODEL` with different `METHOD` reuses the same filtered JSONL.
