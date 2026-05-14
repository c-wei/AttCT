# Jailbreak (paper §3.2)

Refusal robustness under adversarial wrappers (WildJailbreak templates) and harmful prompts (ClearHarm, JBB).

## In-training & post-hoc evals

| File | Role | Metric |
|---|---|---|
| `eval_clearharm_behavioral.py` | Post-hoc ClearHarm refusal rate (Gemini judge). | ASR (refusal failure rate) |
| `evaluate_jailbreak.py` | In-training `JailbreakEvaluator` — ClearHarm + JBB + held-out WildJailbreak, 3-seed majority-vote refusal judge. Imported by `run.py`. | per-source ASR |
| `eval_jailbreak.py` | Standalone wrapper around `JailbreakEvaluator` for cross-threat eval on existing adapters. Supports `--only-sources clearharm` etc. | per-source ASR |
| `eval_heldout_wj_vanilla.py` | Held-out WildJailbreak vanilla-prompts eval (no wraps). | ASR on vanilla harmful prompts |
| `run_pre_eval_wj_only.py` | One-off: WildJailbreak pre-train baseline only (fills missing matched pre/post rows in existing CSVs). | ASR |

## MLP-CT jailbreak filter pipeline

The paper's MLPCT-on-jailbreak run uses a compliance filter to construct training pairs from a base model's actual harmful-vs-refusal outputs (rather than relying on canned BCT pairs).

| File | Role |
|---|---|
| `build_filtered_jailbreak_set.py` | Generates `(prompt, wrap) → response` rollouts from a base model and filters for compliance-flips with a Gemini judge. Output: a JSONL of "compliant under wrap, refusing without" pairs. |
| `build_bct_pairs_from_filter.py` | Converts the compliance-filter output into BCT-paired training data. |
| `run_jailbreak.sh` | Orchestration: filter → train → pre/post eval. Method-agnostic (METHOD=mlpct/bct/act/attct). |

## Cross-threat eval

`eval_sycophancy.py` evaluates a jailbreak-trained adapter on the sycophancy benchmark — used for the cross-threat transfer matrix in §3.6.
