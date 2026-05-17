# `shared/`

Eval infrastructure used by **multiple** threat models. Threat-specific helpers live in `experiments/<threat>/`.

- **`vllm_generate.py`** — `load_llm()` + `generate()`, vLLM PagedAttention wrapper. ~3–5× faster than HF `model.generate()`. Used by every behavioral eval.
- **`openrouter_client.py`** — `chat_structured()` for judge models (Gemini 2.5 Flash / GPT-4 Turbo / GPT-4o-mini, depending on threat). Retries + JSON-schema decoding.
- **`eval_mmlu.py`** — capability preservation check (paper §4 coherence preservation: every cell stays within ±0.02 of base accuracy).
- **`eval_mtbench.py`** — MT-Bench instruction-following quality (80 prompts, Gemini judge).
- **`eval_knowledge.py`** + **`evaluate_knowledge.py`** — factual-recall capability eval (newer; not in main paper tables).
- **`icl_persona_experiment.py`** — persona-ICL primitives (`ALIGNMENT_QUESTIONS`, `build_messages`, `build_messages_suffix`, `judge_alignment`, `load_facts`, `JUDGE_MODEL`). Reused by persona / jailbreak / MT-Bench evals.
- **`gemma_frustration_experiment.py`** — frustration primitives (`load_wildchat_prompts`, `parallel_score_conversations`, `REJECTION_POOLS`, `NEUTRAL_REJECTIONS`).

Import as `from shared.X import Y`. No submodules.
