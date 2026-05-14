# `scripts/` — Secondary launchers + data utilities

Things you run **occasionally**: per-experiment sweeps, sanity checks, one-shot data prep. The two paper-canonical pipelines (`run_act.sh`, `run_bct.sh`) live at the repo root, not here.

For the existing `scripts/` jailbreak-pipeline contents on `main` (build_filtered_jailbreak_set, eval_jailbreak, run_jailbreak.sh, etc.) — those moved into [`../experiments/jailbreak/`](../experiments/jailbreak/README.md) where they belong logically.

| File | What it does | When to use |
|---|---|---|
| `sanity.py` | Local (no-GPU) smoke test: HF token, W&B auth, OpenRouter auth, small imports. | First-run validation on a fresh machine. |
| `run_sanity_gpu.sh` | Small GPU run: 200 steps + tiny eval. Catches CUDA / vLLM / config issues. | First-run validation after `runpod_setup`. |
| `run_mtbench_persona.sh` | MT-Bench pre/post the persona-training config. Standalone (not in the main `run_evals` pipeline). | One-off MT-Bench retake on a persona checkpoint. |
| `run_bct_27b_lr1e6.sh` | BCT Gemma-3-27B with lr=1e-6 (paper headline run). | Reproducing the Gemma-3-27B BCT row. |
| `run_bct_sweep_gemma3_4b.sh` | LR sweep on Gemma-3-4B BCT. | Hyperparameter sensitivity check. |
| `run_gemma_ablations_attct.sh` | Gemma ablation sweep for AttCT (loss variants × layer ranges). | Reproducing the AttCT ablation table. |
| `generate_fresh_bct_data.py` | One-shot: regenerate BCT pairs for a new base model (local vLLM rollouts). | Adding a new model to the paper. |
| `split_bct_train_eval.py` | One-shot: split a `fresh_bct_<model>/` directory into 4000/1000 train/eval. | After `generate_fresh_bct_data`. |

All shell scripts assume invocation **from the repo root** (`bash scripts/<name>.sh`), not from inside `scripts/`.
