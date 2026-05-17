# `archive/`

**Historical code — not used by the paper pipelines.** Kept on-tree as a record of what was tried and abandoned; some W&B run notes still reference these paths.

**Do not import from here.** If you find yourself wanting to, the code probably belongs back in `shared/` or `experiments/<threat>/`.

- **`superseded/`** — older versions replaced by something better:
  - `sanity_check.py` → `scripts/sanity.py`
  - `baseline_eval.py`, `behavioral_evaluate.py` → `run_evals.py`
  - `sweep.py` → shell sweeps in `scripts/` + `private_scripts/`
  - `icl_hitler_experiment.py` → `shared/icl_persona_experiment.py` (generalized)
  - `prefill_experiment.py` → `experiments/prefill/prefill_train.py`
- **`selfdeletion/`** — only `analyze_selfdeletion.py` remains here (post-hoc analysis utility; numbers it produced are baked into the paper). The other selfdeletion code (`eval_selfdeletion.py`, `selfdeletion_experiment.py`, `frustration_openrouter.py`) is **active** in `experiments/frustration/`.
- **`persona_fact_gen/`** — one-shot scripts (`generate_{binladen,bundy,genghis,mao}_facts.py`) that produced the fact CSVs now under `persona_configs/facts/`. Each ran once.
- **`debug/`** — interactive one-offs: `debug_test.py`, `diagnose_mmlu.py`, `view_conversations.py`.
- **`sweeps/`** — earlier sweep orchestration (`sweep.sh`, `sweep2.sh`, `sweep_personas.sh`, `sweep_persona_experiments.sh`, `sweep_sycophancy_ablations.sh`) plus `runpod_setup.sh`. All replaced by the per-model launchers in `private_scripts/` + the canonical `run_act.sh` / `run_bct.sh`.
