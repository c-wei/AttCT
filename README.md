# AttCT — Attention Consistency Training

Companion code for the ACL paper *Attention Consistency Training: Robustness Across Threat Models via Hidden-State Consistency*.

The paper studies **4 consistency-training methods** — Attention Consistency Training (AttCT), Behavioral Consistency Training (BCT), MLP Consistency Training (MLPCT), and Activation Consistency Training (ACT) — applied to **5 robustness threat models** across **5 instruction-tuned models**.

| Methods | Threat models | Models |
|---|---|---|
| AttCT — JSD on attention weights | Sycophancy | Gemma-3-4B-IT |
| BCT — SFT on clean responses to wrapped prompts | Jailbreak | Gemma-3-27B-IT |
| MLPCT — cosine on MLP post-activations | Persona ICL (44 personas) | Llama-3.1-8B-Instruct |
| ACT — corrected Irpan-et-al activation consistency | Prefill attacks | Qwen3-4B-Instruct-2507 |
|  | Frustration / self-deletion rollouts | Qwen3-8B |

## Quickstart

```bash
# Sycophancy + jailbreak + persona + frustration + MMLU + MT-Bench, AttCT path
bash run_act.sh --config configs/act_sycophancy_llama31_8b_v2.yaml

# Same pipeline, BCT path
bash run_bct.sh --config configs/bct_lora_gemma3_27b.yaml

# Prefill (separate threat with its own trainer + eval)
bash experiments/prefill/prefill_train.sh
bash experiments/prefill/run_prefill_eval_custds.sh
```

Both pipelines run pre-evals → training → post-evals through `run_evals.py` (one shared vLLM session per phase). Skip flags: `--skip-pre-evals`, `--skip-training`, `--skip-rollouts`. See [`scripts/README.md`](scripts/README.md) for secondary launchers.

## Repository layout

| Path | Contents |
|---|---|
| `run.py` | Main config-driven training entry. Selects loss + trainer based on YAML. |
| `train.py`, `interleaved_trainer.py` | Trainer classes (plain, BCT, interleaved AttCT+KL, intelligence). |
| `run_evals.py` | Unified eval orchestrator — runs every behavioral eval in one vLLM session. |
| `evaluate.py`, `hooks.py` | In-training consistency-loss eval and MLP-state capture hook. |
| `run_act.sh`, `run_bct.sh` | Paper-canonical pre-eval → train → post-eval pipelines. |
| [`losses/`](losses/README.md) | Loss-function implementations (AttCT, ACT, MLPCT, JSD variants, SFT/BCT). |
| [`data/`](data/README.md) | `AttCTDataset`, prompt wrappers, KL-regularization dataloader. |
| [`shared/`](shared/README.md) | Cross-cutting eval infra: vLLM helper, OpenRouter judge client, MMLU / MT-Bench / Knowledge evals, persona-ICL helpers. |
| [`experiments/`](experiments/README.md) | Per-threat training scripts, behavioral evals, and metric implementations. |
| [`scripts/`](scripts/README.md) | Secondary launchers + one-shot data utilities + local sanity check. |
| [`configs/`](configs/README.md) | YAML configs for each (method, model, threat) combination. |
| [`persona_configs/`](persona_configs/README.md) | Fact files + YAML for the 5 named ICL personas. |
| [`datasets/`](datasets/README.md) | On-disk data assets (BCT pairs, prefill seeds, frustration prompts). |
| [`findings/`](findings/README.md) | Lab-notebook write-ups for each numbered paper result. |
| [`private_scripts/`](private_scripts/README.md) | Per-model "best run" launchers used to reproduce the paper headline numbers. |
| [`results/`](results/README.md) | Run outputs (gitignored except `legacy/` and `attention_viz/`). |
| [`tests/`](tests/README.md) | Pytest-discovered tests. Run from repo root: `pytest`. |
| [`archive/`](archive/README.md) | **Historical code, not used by the paper.** Kept for reference. |

## Results & findings

Headline numbers for each (method, threat) cell, with W&B run IDs and full per-model breakdowns, live in [`findings/`](findings/README.md). The main paper figure summarising the cross-threat matrix is reproduced from data in `findings/act_v2_results.md` and `findings/bct_gemma3_27b_lora_findings.md`.

## Environment

This project uses `uv`. Run any script via `uv run python …` or `uv run --env-file .env python …` (the OpenRouter / W&B keys live in `.env`). For RunPod GPU setup notes, see `archive/sweeps/runpod_setup.sh` (no longer the canonical setup path — replaced by the per-model launchers in `private_scripts/`).
