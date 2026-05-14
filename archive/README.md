# `archive/` — Historical code, **not used by the paper pipelines**

Everything here is **legacy / superseded / orphaned**. Kept on-tree as a record of what was tried and abandoned, and because some of these files are still referenced in old W&B run notes.

**Do not import from `archive/`.** If you find yourself wanting to, the code probably belongs back in `shared/` or `experiments/<threat>/`. Move it out of archive deliberately.

## `superseded/` — older versions of code now done better elsewhere

| File | Replaced by |
|---|---|
| `sanity_check.py` | `scripts/sanity.py` (unified local + GPU scope) |
| `baseline_eval.py` | `run_evals.py` (unified eval pipeline) |
| `behavioral_evaluate.py` | `run_evals.py` (vLLM-based behavioral evals) |
| `sweep.py` | `scripts/*.sh` + `private_scripts/*.sh` (config-driven shell sweeps) |
| `icl_hitler_experiment.py` | `shared/icl_persona_experiment.py` (generalized to all personas) |
| `prefill_experiment.py` | `experiments/prefill/prefill_train.py` (unified prefill trainer for all 4 methods) |

## `selfdeletion/` — pure orphans

| File | Why archived |
|---|---|
| `analyze_selfdeletion.py` | Post-hoc analysis utility; not imported anywhere. Numbers it produced are baked into `findings/selfdeletion_findings.md`. |

(The other 3 selfdeletion files — `eval_selfdeletion.py`, `selfdeletion_experiment.py`, `frustration_openrouter.py` — live in `experiments/frustration/` because they are imported by the active rollout pipeline.)

## `persona_fact_gen/` — one-shot data generators

`generate_{binladen,bundy,genghis,mao}_facts.py` — generated the biographical Q&A fact files now committed under `persona_configs/facts/`. The scripts ran once each (via OpenRouter against Gemini) and are kept as a historical record.

## `debug/` — interactive one-offs

| File | What it was for |
|---|---|
| `debug_test.py` | Ad-hoc debugging script. |
| `diagnose_mmlu.py` | MMLU result diagnostic (compared logprob extraction against generation). |
| `view_conversations.py` | Transcript JSONL inspector. |

## `sweeps/` — earlier sweep orchestration + RunPod setup

| File | Purpose |
|---|---|
| `sweep.sh`, `sweep2.sh` | Original bash sweep harnesses (looped over configs). |
| `sweep_personas.sh`, `sweep_persona_experiments.sh` | Persona-sweep variants. |
| `sweep_sycophancy_ablations.sh` | Sycophancy loss-variant ablation sweeps. |
| `runpod_setup.sh` | First-version RunPod GPU instance setup. Replaced by the per-model launchers in `private_scripts/`. |
