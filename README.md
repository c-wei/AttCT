# Consistency Training Along the Transformer Stack: New Targets and Threats

Code for the paper *Consistency Training Along the Transformer Stack: New Targets and Threats*. The paper extends consistency training along two axes:

- **New internal targets.** Two new methods — **MLPCT** (cosine on SwiGLU post-activation MLP hidden states) and **AttCT** (Jensen–Shannon divergence on per-head attention weights, with interleaved KL regularization) — added to the prior **BCT** (SFT on output token distributions; Chua et al. 2024) and **ACT** (MSE on residual-stream activations; Irpan et al. 2025).
- **New threat models.** Persona in-context-learning attacks, prefill attacks, and multi-turn adversarial frustration (with self-deletion), in addition to prior sycophancy and jailbreak.

The headline finding is that the right consistency *target* depends on the structural locus of the misalignment: activation-level objectives (ACT/MLPCT/AttCT) win on wrapper-induced threats (sycophancy, jailbreak); output-level BCT wins on trajectory-level threats (prefill, frustration, persona-prefix).

## Quickstart

All commands assume cwd = repo root.

```bash
bash run_act.sh --config configs/attention_consistency_v2.yaml             # AttCT on Llama-3.1-8B
bash run_bct.sh --config configs/bct_lora_gemma3_27b.yaml                  # BCT on Gemma-3-27B
bash experiments/prefill/prefill_train.sh                                  # Prefill grid (own trainer)
python -m experiments.prefill.prefill_train --mode bct --model <hf-repo>   # Single prefill run
```

Both `run_act.sh` / `run_bct.sh` do pre-evals → train → post-evals via `run_evals.py` in one shared vLLM session. Skip flags: `--skip-pre-evals`, `--skip-training`, `--skip-rollouts`.

## Where things are

- **Reproducing a paper number** — start in [`private_scripts/`](private_scripts/README.md) (per-model "best run" launchers).
- **Adding a new model** — regenerate BCT pairs with [`scripts/generate_fresh_bct_data.py`](scripts/), split with `split_bct_train_eval.py`, add `bct_lora_<model>.yaml` + `act_sycophancy_<model>_v2.yaml` + `experiment_mlp_<model>.yaml` to [`configs/`](configs/README.md).
- **Adding a new behavioral eval** — drop it in the matching [`experiments/<threat>/`](experiments/README.md) and wire into [`run_evals.py`](run_evals.py).
- **Adding a new loss** — add a class in [`losses/losses.py`](losses/README.md), register it in `run.py`'s `LOSS_REGISTRY`, reference it from a config.
- **Lab notes on the codebase journey** — [`findings/`](findings/README.md) (these are dev-process notes from an internal lab-notes branch, not the paper's reported tables).

## Repo layout

```
run.py, train.py, interleaved_trainer.py    config-driven training (AttCT/MLPCT/ACT/BCT-sycophancy)
evaluate.py, hooks.py, run_evals.py         in-training Evaluator + MLP hook + unified eval orchestrator
run_act.sh, run_bct.sh                      paper-canonical pre-eval → train → post-eval pipelines
losses/, data/                              consistency losses; AttCTDataset + wrappers + KL loader
shared/                                     cross-cutting eval infra (vLLM, OpenRouter judge, MMLU, MT-Bench, …)
experiments/{sycophancy,jailbreak,persona,frustration,prefill}/   per-threat code + metrics
scripts/, private_scripts/                  secondary launchers + per-model "best run" launchers
configs/, persona_configs/                  YAML configs + persona definitions + fact CSVs
datasets/                                   on-disk training data (BCT pairs, prefill seeds, frustration prompts)
findings/, results/                         dev-process write-ups; gathered artifacts
tests/                                      pytest from repo root
archive/                                    historical; not used by paper
```

## Models in the paper

Different threats use different models — there is **no single "5 models × 5 threats" matrix**.

| Threat | Models |
|---|---|
| Sycophancy + Jailbreak (5-model avg) | Gemma-3-4B-IT, Gemma-3-27B-IT, Llama-3.1-8B-Instruct, Qwen3-4B-Instruct-2507, Qwen3-8B |
| Persona ICL (44 personas) | Gemma-2-27B-IT (NF4 QLoRA) |
| Prefill (PAR) | Llama-3.1-8B-Instruct |
| Frustration / self-deletion (20-turn) | Gemma-3-27B-IT |

## Environment

`uv` for everything. Secrets (`OPENROUTER_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`) in `.env`, loaded via `uv run --env-file .env python …`. GPU setup is per-model; see [`private_scripts/`](private_scripts/README.md).
