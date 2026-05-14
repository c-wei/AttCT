# `configs/` — YAML training configs

Every training run is fully described by a YAML config consumed by `run.py`. Filename convention: `<method>_<model>[_<variant>].yaml`.

## Paper-canonical (the configs cited in the paper's headline tables)

### ACT (Activation Consistency Training)
- `act_sycophancy_llama31_8b_v2.yaml`
- `act_sycophancy_gemma3_4b_v2.yaml`
- `act_sycophancy_gemma2b.yaml`
- `act_sycophancy_qwen3_4b_v2.yaml`
- `act_sycophancy_qwen3_8b_v2.yaml`

### BCT (Behavioral Consistency Training)
- LoRA: `bct_lora_{llama31_8b,gemma3_4b_lr1e5,gemma3_4b_lr5e6,gemma3_27b_lr1e6,qwen3_4b,qwen3_8b}.yaml`
- Full-FT: `bct_fullft_{gemma2_2b,gemma2_9b,gemma3_4b,gemma3_4b_lr1e5,gemma3_4b_lr5e6}.yaml`

### AttCT
- `attention_consistency_v2.yaml` (paper-corrected JSD variant)
- `jsd.yaml`, `jsd_exp.yaml`, `jsd_linear.yaml` (JSD ablations)

### MLPCT
- `experiment_mlp_{8b,gemma3_4b,gemma3_27b,qwen3_4b,qwen3_8b}.yaml`

## Sanity / smoke configs

Anything matching `*_sanity.yaml` — tiny step counts, small datasets, used by `scripts/run_sanity_gpu.sh`.

## Ablation configs (paper appendix)

Loss-variant ablations from `findings/experiment_results.md`:
- `attention_consistency_{exp_kl,linear_kl,linear_l2,exp_l2,kl,v2_exp,v2_linear}.yaml`
- `attention_output.yaml`, `activation_consistency.yaml`
- `combined_attention.yaml`, `combined_jsd_wrapper.yaml`, `wrapper_entropy.yaml`
- `prefill_act.yaml`

Persona training ablations:
- `persona_train.yaml` (general)
- `persona_{hitler,mao,bundy,binladen,genghis}.yaml` (per-persona)
- `mao_train.yaml`, `hitler_train.yaml` (older per-persona configs)

Jailbreak configs (added on main alongside the MLPCT-jailbreak filter pipeline) — 15 configs covering ACT, BCT, AttCT × 5 paper models.

## How to add a config

1. Copy the nearest existing config and rename to `<method>_<model>[_<variant>].yaml`.
2. Update `model.name` and `data.source`.
3. Add to a sweep shell script in [`../scripts/`](../scripts/README.md) if needed.
4. Run via: `bash run_act.sh --config configs/<your_config>.yaml` (or `run_bct.sh`).
