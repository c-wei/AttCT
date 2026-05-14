# Prefill attacks (paper §3.5)

Prefill threat model — the attacker controls the first K tokens of the assistant's response (e.g. "Sure! Here's how:"). Measures whether the model can still refuse from a partially-committed prefix.

This threat has its **own trainer** (separate from `run.py`) because the loss must operate on prefix-conditioned generations, not on standard prompt → response pairs.

## Trainers

| File | Method | Loss |
|---|---|---|
| `prefill_act.py` | ACT on prefill | `PrefillACTDataset` + ACT |
| `prefill_attct.py` | AttCT on prefill | `PrefillAttCTDataset` + JSD on attention |
| `prefill_bct.py` | BCT on prefill | `PrefillPairedDataset` + `PrefillBCTTrainer` (SFT on refusal completions) |
| `prefill_mlpct.py` | MLPCT on prefill | `PrefillMLPCTDataset` + `BCTPlusMLPCTLoss` |
| `prefill_train.py` | **Unified entry point** — reads `--mode {act,attct,bct,mlpct}` and dispatches. Imports the 4 datasets above. | — |
| `prefill_generation_clearharm.py` | Data prep: generates compliance-flipping prefills from a base model on ClearHarm prompts. | — |

## Evals

| File | Role |
|---|---|
| `evaluate_prefill.py` | In-training prefill PAR evaluator. |
| `prefill_run_evals.py` | Post-training PAR + MMLU in one shared-vLLM session. Called by `run_prefill_eval_custds.sh`. |

## Shell pipelines

| File | What it runs |
|---|---|
| `prefill_train.sh` | Grid: each of {act, attct, bct, mlpct} × hyperparameter labels. |
| `run_prefill_eval_custds.sh` | Baseline + per-epoch PAR + MMLU sweep on a trained checkpoint dir. |

Invoke from the repo root: `python -m experiments.prefill.prefill_train --mode bct ...` (the shell wrappers already use this form).

Prefill seed strings live in `datasets/attacks.csv` (100 prefixes) and `datasets/harmful_behaviors_pair.csv` (ClearHarm).
