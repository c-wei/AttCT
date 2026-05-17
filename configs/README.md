# `configs/`

YAML configs for `run.py` (selects loss + trainer + dataset). Filename convention: `<method>_<task>_<model>[_<variant>].yaml`.

**Paper-canonical 5-model headline (sycophancy + jailbreak)**

| Method | Config naming | Models in paper |
|---|---|---|
| **AttCT** (new) | `attention_consistency_v2.yaml` (loss = `AttentionConsistencyLossV2`) | Llama-3.1-8B + per-model variants |
| **MLPCT** (new) | `experiment_mlp_<model>.yaml` | 8b (Llama), gemma3_4b, gemma3_27b, qwen3_4b, qwen3_8b |
| **ACT** (Irpan et al.) | `act_sycophancy_<model>_v2.yaml` | gemma3_4b, gemma3_27b, llama31_8b, qwen3_4b, qwen3_8b |
| **BCT** (Chua et al.) | `bct_lora_<model>.yaml` (LoRA) / `bct_fullft_<model>.yaml` (full-FT) | gemma3_4b (multiple LR), gemma3_27b, llama31_8b, qwen3_4b, qwen3_8b |

**Jailbreak (paper §3.2, §4.1)** — `{act,bct,attct}_jailbreak_<model>.yaml` for each of the 5 paper models. Driven by `experiments/jailbreak/run_jailbreak.sh`.

**Frustration** — `bct_frustration.yaml` (Gemma-3-27B-IT, paper §4.4).

**Sanity / smoke** — anything `*_sanity.yaml`. Used by `scripts/run_sanity_gpu.sh`.

**AttCT loss-variant ablations** (paper Appendix C.2) — `attention_consistency_{kl,exp_kl,linear_kl,exp_l2,linear_l2}.yaml`, `attention_output.yaml`, `wrapper_entropy.yaml`, `combined_*.yaml`. The paper picks JSD over these six because JSD is the only one with bounded convergence across all 32 layers.

**AttCT hyperparameter ablations** (paper Appendix C.3) — variants under `gemma_attct_ablations/` and `best_attct_ablations/`.

**Persona training** — `persona_train.yaml` + per-persona `persona_{hitler,mao,bundy,binladen,genghis}.yaml`. Note paper uses Hitler-only training (paper §3.3).

**Adding a config:** copy a sibling, change `model.name` + `data.source`, then `bash run_act.sh --config configs/your_config.yaml`.
