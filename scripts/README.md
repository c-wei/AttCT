# `scripts/`

Secondary launchers, sanity checks, and one-shot data prep. The paper-canonical pipelines (`run_act.sh`, `run_bct.sh`) live at the repo root.

All scripts assume cwd = repo root (`bash scripts/<name>.sh`).

**Sanity / smoke**
- `sanity.py` — local (no-GPU) check: HF + W&B + OpenRouter auth, small import smoke.
- `run_sanity_gpu.sh` — GPU smoke: 200 steps + tiny eval per model. Catches CUDA / vLLM / config issues.

**Per-experiment launchers** (paper appendix runs and ablations)
- `run_bct_27b_lr1e6.sh` — BCT on Gemma-3-27B with lr=1e-6 (LR sensitivity check).
- `run_bct_sweep_gemma3_4b.sh` — LR sweep on Gemma-3-4B BCT.
- `run_gemma_ablations_attct.sh` — 13-cell AttCT ablation grid (LoRA targets × layer weights × layer selection × rank × KL interleaving, paper Appendix C.3 table).
- `run_mtbench_persona.sh` — MT-Bench retake on a persona-training checkpoint.

**Data prep** (run once per new model)
- `generate_fresh_bct_data.py` — regenerate sycophancy BCT pairs for a new base model (local vLLM rollouts).
- `split_bct_train_eval.py` — produce 4000/1000 train/eval splits in a `datasets/fresh_bct_<model>/`.
