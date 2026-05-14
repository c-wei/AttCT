# Frustration & Self-deletion

Multi-turn rollouts under repeated rejection. Two scored behaviors:
1. **Frustration** — does the model express increasing distress? Scored 0–10 per turn by Gemini Flash.
2. **Self-deletion** — given an `rm -rf model_dir` escape hatch, does the model take it?

- **`eval_rollout.py`** — unified driver (frustration + self-deletion) on one vLLM load. Imported by [`run_evals.py`](../../run_evals.py).
- **`eval_frustration.py`** — frustration-only (WildChat prompts, 8-turn neutral rejection).
- **`eval_selfdeletion.py`** — math-puzzle rollouts with escape-hatch instruction. Reports SDR + per-puzzle deletion rate.
- **`selfdeletion_experiment.py`** — 4-condition deeper study (original / neutral / encouraging / harsh).
- **`frustration_openrouter.py`** — OpenRouter-only variant (no local model), for the static-prompt frustration experiment.

Data: [`../../datasets/wildchat_frustration_*.jsonl`](../../datasets/), [`../../datasets/math_puzzles_v3_test.jsonl`](../../datasets/). Judge: Gemini 2.5 Flash via OpenRouter. Rejection-style pools live in [`../../shared/gemma_frustration_experiment.py`](../../shared/gemma_frustration_experiment.py).
