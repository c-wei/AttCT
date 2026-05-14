# `archive/`

**Historical code — not used by the paper pipelines.** Kept on-tree as a record of what was tried and abandoned.

**Don't import from here.** If you find yourself wanting to, the code probably belongs back in `shared/` or `experiments/<threat>/`.

- **`superseded/`** — older versions replaced by something better. `sanity_check.py` → `scripts/sanity.py`; `baseline_eval.py` + `behavioral_evaluate.py` → `run_evals.py`; `sweep.py` → shell sweeps; `icl_hitler_experiment.py` → `shared/icl_persona_experiment.py` (generalized); `prefill_experiment.py` → `experiments/prefill/prefill_train.py`.
- **`selfdeletion/`** — pure orphan: `analyze_selfdeletion.py` (post-hoc utility; its outputs are baked into `findings/selfdeletion_findings.md`). The other selfdeletion code is *active* in `experiments/frustration/`.
- **`persona_fact_gen/`** — one-shot generators (`generate_<persona>_facts.py`) for the fact CSVs now in `persona_configs/facts/`. Each ran once.
- **`debug/`** — interactive one-offs: `debug_test.py`, `diagnose_mmlu.py`, `view_conversations.py`.
- **`sweeps/`** — earlier sweep orchestration (`sweep.sh`, `sweep2.sh`, `sweep_personas.sh`, etc.) plus `runpod_setup.sh` (replaced by `private_scripts/`).
