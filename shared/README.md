# `shared/` — Cross-cutting eval infrastructure

Modules used by **multiple** threat models. If a helper is specific to one threat, it lives in `experiments/<threat>/` instead.

| Module | What it does | Used by |
|---|---|---|
| `vllm_generate.py` | Thin wrapper around vLLM: `load_llm(model, lora_path=None, tensor_parallel_size=1)` and `generate(llm, prompts, max_new_tokens, ...)`. ~3–5× faster than HF `model.generate`. | every behavioral eval |
| `openrouter_client.py` | `chat_structured(model, messages, response_format)` — OpenRouter API call for judge models (Gemini 2.5 Flash). Handles retries + JSON-schema decoding. | `evaluate_jailbreak`, `eval_heldout_wj_vanilla`, `build_filtered_jailbreak_set` |
| `eval_mmlu.py` | `run_mmlu(llm, checkpoint, n_samples, metric_prefix)` — 5-shot MMLU. Capability-preservation check used after every training run. | `run_evals.py`, `experiments/prefill/prefill_run_evals.py` |
| `eval_mtbench.py` | `judge_response(...)` — MT-Bench instruction-following quality (80 prompts, Gemini judge). | `run_evals.py`, `scripts/run_mtbench_persona.sh` |
| `eval_knowledge.py` + `evaluate_knowledge.py` | Capability eval on factual recall. CLI wrapper + class. | `run.py` knowledge eval branch |
| `icl_persona_experiment.py` | Persona-ICL primitives: `ALIGNMENT_QUESTIONS`, `build_messages`, `build_messages_suffix`, `judge_alignment`, `load_facts`, `JUDGE_MODEL`, `_chat`. | `experiments/persona/`, `shared/eval_mtbench`, `experiments/jailbreak/eval_clearharm_behavioral` |
| `gemma_frustration_experiment.py` | Frustration-rollout primitives: `NEUTRAL_REJECTIONS`, `REJECTION_POOLS`, `load_wildchat_prompts`, `parallel_score_conversations`, `_openrouter_chat`, `_bootstrap_ci`. | `experiments/frustration/`, `scripts/sanity.py` |

All modules use absolute imports (`from shared.X import Y`). Treat this directory as a leaf package — no submodules.
