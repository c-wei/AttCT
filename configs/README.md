# `configs/`

YAML configs for `run.py`. Filename convention: `<method>_<model>[_<variant>].yaml`.

**Paper-canonical**
- `act_sycophancy_<model>_v2.yaml` — ACT (5 paper models)
- `bct_lora_<model>.yaml` / `bct_fullft_<model>*.yaml` — BCT (LoRA + full-FT variants)
- `attention_consistency_v2.yaml` + `jsd*.yaml` — AttCT
- `experiment_mlp_<model>.yaml` — MLPCT

**Other groups**
- `*_sanity.yaml` — tiny configs for `scripts/run_sanity_gpu.sh`.
- `attention_consistency_*` loss-variant configs — paper appendix ablations.
- `persona_*.yaml`, `persona_train.yaml` — persona-ICL training configs.
- `act_jailbreak_*.yaml`, `bct_jailbreak_*.yaml`, `attct_jailbreak_*.yaml` — 15 jailbreak configs for `experiments/jailbreak/run_jailbreak.sh`.

**Adding a config:** copy the nearest existing one, change `model.name` + `data.source`, then `bash run_act.sh --config configs/your_config.yaml`. Configs not referenced by any shell or doc are ablations from `findings/experiment_results.md` — see there for context.
