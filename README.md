# AttCT — Attention Consistency Training

Code for the ACL paper *Attention Consistency Training: Robustness Across Threat Models via Hidden-State Consistency*. Four consistency-training methods (AttCT, BCT, MLPCT, ACT) × five threat models (sycophancy, jailbreak, persona ICL, prefill, frustration) × five instruction-tuned models (Gemma-3-4B/27B, Llama-3.1-8B, Qwen3-4B/8B).

## Quickstart

```bash
bash run_act.sh --config configs/act_sycophancy_llama31_8b_v2.yaml   # AttCT path
bash run_bct.sh --config configs/bct_lora_gemma3_27b.yaml            # BCT path
bash experiments/prefill/prefill_train.sh                            # Prefill (separate trainer)
```

Both `run_act.sh` / `run_bct.sh` do pre-evals → train → post-evals via `run_evals.py` (one shared vLLM session per phase). All scripts assume invocation from the repo root.

## Where to look

- **Reproducing a paper number?** → [`findings/`](findings/README.md) for the per-row write-ups, then the matching launcher in [`private_scripts/`](private_scripts/README.md).
- **Adding a new model?** → [`scripts/generate_fresh_bct_data.py`](scripts/) + a new YAML in [`configs/`](configs/README.md).
- **Adding a new eval?** → drop it in the matching [`experiments/<threat>/`](experiments/README.md) and wire it into [`run_evals.py`](run_evals.py).
- **Adding a new consistency loss?** → [`losses/`](losses/README.md), then reference it in a config.
- **Debugging an import?** — packages are `shared.*`, `experiments.<threat>.*`, `data.*`, `losses.*`. Run from repo root.

## Layout

```
run.py, train.py, interleaved_trainer.py     ← config-driven training
evaluate.py, hooks.py, run_evals.py          ← in-training + behavioral eval
run_act.sh, run_bct.sh                       ← paper-canonical pipelines
losses/, data/                               ← what to optimize, what to feed
shared/                                      ← cross-cutting eval infra
experiments/{sycophancy,jailbreak,persona,frustration,prefill}/
scripts/, private_scripts/                   ← launchers (secondary, per-model)
configs/, persona_configs/, datasets/        ← inputs
findings/, results/                          ← outputs (committed write-ups, raw artifacts)
tests/                                       ← `pytest` from repo root
archive/                                     ← historical; not used by the paper
```

## Environment

`uv` for everything. Secrets (`OPENROUTER_API_KEY`, `WANDB_API_KEY`, `HF_TOKEN`) in `.env`, loaded via `uv run --env-file .env python …`. GPU setup is per-model; see [`private_scripts/`](private_scripts/README.md).
