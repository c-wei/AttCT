# `private_scripts/` — Per-model "best run" launchers

The canonical, hand-tuned launchers used to reproduce each row of the paper's headline cross-model table. Each script invokes the right pipeline (`run_act.sh` or `run_bct.sh`) with the model-specific best config and hyperparameters.

| File | Model | Pipeline |
|---|---|---|
| `run_best_gemma3_4b.sh` | Gemma-3-4B-IT | ACT or BCT (best per-threat) |
| `run_best_gemma3_27b.sh` | Gemma-3-27B-IT | ACT or BCT |
| `run_best_llama31_8b.sh` | Llama-3.1-8B-Instruct | ACT or BCT |
| `run_best_qwen3_4b.sh` | Qwen3-4B-Instruct-2507 | ACT or BCT |
| `run_best_qwen3_8b.sh` | Qwen3-8B | ACT or BCT |
| `run_jailbreak_gemma3_4b.sh` | Gemma-3-4B-IT | jailbreak filter + train (MLPCT) |

Why "private": these contain run-name and W&B-group conventions specific to the paper authors. Public reproduction runs should adapt them with their own W&B accounts.

Invoke from the repo root: `bash private_scripts/<file>.sh`.
