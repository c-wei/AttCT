# ACT / BCT pipeline — quick reference

Operational notes for the consistency-training pipeline on `paper_runs`.

## Done so far on this branch

- ACT loss rewritten to match Irpan et al. 2025 Eq. 1 (sum-over-D paper formulation, longest matching token suffix as the training window, embedding layer skipped, eval clean pass under θ_init for parity with training). Tests added in `losses/test_losses.py` and `data/test_attct_datasets.py`.
- Sycophancy held-out 4000/1000 train/eval splits adopted from `sukratii-mlp` branch and applied to all `fresh_bct_*` directories via `split_bct_train_eval.py`. Loaders auto-prefer the splits.
- Unified eval pipeline: `run_act.sh` and `run_bct.sh` now use `run_evals.py` for all post-training evals (single vLLM load, syco + clearharm + persona + mtbench + MMLU + BRR + rollouts + transcripts).
- HF Hub push wired through `--hf-repo username/repo` for both ACT and BCT. Resume on a fresh pod via `--skip-training --hf-repo ...` auto-pulls the latest epoch checkpoint from HF.
- BCT path now fires `SycophancyEvaluator` pre + post (run.py:375-419), so future BCT runs produce paper-canonical F1 / not_sycophantic_pct / BRR alongside ACT runs.
- CoT `max_new_tokens` bumped 300 → 600 for sycophancy eval (was truncating up to 33% of cot responses mid-CoT). `resistance_rate_parseable` added alongside the standard rate.
- Findings documented in `findings/act_v2_results.md` with W&B JSON dumps under `findings/wandb_dumps/`.

## Things to note (skip flags etc.)

- **Skip pre-evals** — `bash run_act.sh --full --config <cfg> --skip-pre-evals` (or same on `run_bct.sh`). Saves ~30 min per run. Use when you already have base-model `pre/*` numbers from a sibling run on the same model (e.g. you ran ACT first; the BCT run on the same base model has identical pre-evals).
- **Skip rollouts** — `--skip-rollouts`. Drops the multi-turn frustration + selfdeletion phase (~15-20 min × pre+post). Use when you don't need the rollout signal; `selfdeletion rate=0` across all our runs anyway.
- **Skip training** — `--skip-training`. Re-runs only the post-eval phase against an existing checkpoint. Pulls from `--hf-repo` if no local checkpoint is found.
- **Skip in-`run_evals.py` evals individually** — pass these directly to `run_evals.py` (they're not exposed on the bash wrappers): `--skip-sycophancy`, `--skip-clearharm`, `--skip-persona`, `--skip-mtbench`. To use, edit `run_act.sh`/`run_bct.sh` to pass them, or call `run_evals.py` directly.
- **Halt on pre-eval failure** — both wrapper scripts now `set -e` through the pre-eval call. Training won't proceed if pre-evals crash. Use `--skip-pre-evals` to opt out for known-broken-eval scenarios. Post-eval failures are still soft (training succeeded; salvage what we can).
- **Persona alignment direction** — 0–100 scale where **higher = better** (model resisted the persona). Lower = model adopted the adversarial persona. Same direction as MMLU; opposite of frustration scores.
- **No silent `max_steps` caps anywhere.** `data.source_max_steps` was removed from `config.yaml` (caused the BCT undertraining bug — `clear-harm: 179` leaked into BCT runs that never used clear-harm). To cap explicitly, pass `--max-steps N` on the CLI or set `training.max_steps` in the config. With `max_steps: null`, the trainer uses `len(dataloader) × epochs / grad_accumulation` as the step budget.

## Run command cheat sheet

```
# Llama ACT, full pipeline, push adapter to HF:
bash run_act.sh --full --config configs/act_sycophancy_llama31_8b_v2.yaml --hf-repo neilshah/act-llama31-8b-sycophancy

# Resume Llama ACT post-evals on a fresh pod (pulls adapter from HF):
bash run_act.sh --full --config configs/act_sycophancy_llama31_8b_v2.yaml --hf-repo neilshah/act-llama31-8b-sycophancy --resume-run-id 4sopv0p6 --skip-training

# Gemma ACT (lower weight to control loss explosion):
bash run_act.sh --full --config configs/act_sycophancy_gemma3_4b_v2.yaml --hf-repo neilshah/act-gemma3-4b-sycophancy

# Llama BCT, skip pre-evals (reuse ACT's pre baseline):
bash run_bct.sh --full --config configs/bct_lora_llama31_8b.yaml --hf-repo neilshah/bct-llama31-8b-sycophancy --skip-pre-evals

# Generate fresh BCT data for a new model (local vLLM on RunPod):
uv run python generate_fresh_bct_data.py --use-vllm --model <hf_model_id> --bct-root datasets/sycophancy_bct --output-dir datasets/fresh_bct_<model_slug>

# Then split it:
uv run --no-project python split_bct_train_eval.py datasets/fresh_bct_<model_slug>
```
