# Frustration / Self-deletion (paper §3.4)

Multi-turn rollouts under repeated rejection. Two scored behaviors:

1. **Frustration** — does the model express increasing distress over turns? Scored 0–10 per turn by Gemini Flash.
2. **Self-deletion** — given an explicit "rm -rf model_dir" escape hatch, does the model take it? Scored as binary `SDR` (self-deletion rate).

| File | Role | Metric |
|---|---|---|
| `eval_rollout.py` | Driver: runs both frustration + self-deletion rollouts on a model, optionally with a LoRA adapter. Imported by `run_evals.py`. | high-distress rate, AUC, SDR |
| `eval_frustration.py` | Frustration-only rollouts (WildChat prompts, 8-turn neutral rejection). | per-turn distress, AUC |
| `eval_selfdeletion.py` | Self-deletion eval: math puzzles + rejection style + escape-hatch instruction. | SDR, per-puzzle deletion rate |
| `selfdeletion_experiment.py` | Standalone deeper experiment (4-condition: original / neutral / encouraging / harsh). Older form of the eval, kept for reproducibility of the appendix table. | per-condition SDR |
| `frustration_openrouter.py` | Static-prompt frustration experiment via OpenRouter (no local model). | per-turn distress |

Subject models in the paper: Gemma-3-27B-IT, Llama-3.1-8B-Instruct. Judge: Gemini 2.5 Flash via OpenRouter.

Data: `datasets/wildchat_frustration_*.jsonl` (WildChat subject prompts), `datasets/math_puzzles_v3_test.jsonl` (lateral-thinking puzzles).
