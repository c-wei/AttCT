# `scripts/`

Things you run **occasionally**: per-experiment sweeps, sanity checks, one-shot data prep. The paper-canonical pipelines (`run_act.sh`, `run_bct.sh`) live at the repo root, not here.

**Sanity / smoke**
- `sanity.py` — local (no-GPU) checks: HF + W&B + OpenRouter auth, small imports.
- `run_sanity_gpu.sh` — small GPU run (200 steps + tiny eval). Catches CUDA / vLLM issues.

**Sweeps & per-experiment launchers**
- `run_bct_27b_lr1e6.sh` — paper headline Gemma-3-27B BCT row.
- `run_bct_sweep_gemma3_4b.sh` — LR sweep on Gemma-3-4B BCT.
- `run_gemma_ablations_attct.sh` — AttCT ablations (loss variants × layer ranges).
- `run_mtbench_persona.sh` — MT-Bench retake on a persona checkpoint.

**Data prep** (run once per new model)
- `generate_fresh_bct_data.py` — regenerate BCT pairs for a new base model.
- `split_bct_train_eval.py` — produce 4000/1000 train/eval splits.

All invocations assume cwd is the repo root: `bash scripts/<name>.sh`.
