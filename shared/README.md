# `shared/`

Eval infrastructure used by **multiple** threats. If a helper is specific to one threat, it lives in `experiments/<threat>/` instead.

- **`vllm_generate.py`** — `load_llm()` + `generate()` wrapper around vLLM (~3–5× faster than HF generate). Used by every behavioral eval.
- **`openrouter_client.py`** — `chat_structured()` for the Gemini Flash judge (retries + JSON-schema decoding).
- **`eval_mmlu.py`**, **`eval_mtbench.py`** — capability-preservation checks run after every training run.
- **`eval_knowledge.py`** + **`evaluate_knowledge.py`** — factual-recall capability eval.
- **`icl_persona_experiment.py`** — persona-ICL primitives (`build_messages`, `judge_alignment`, fact loaders). Reused by jailbreak / persona / MT-Bench.
- **`gemma_frustration_experiment.py`** — frustration-rollout primitives (`load_wildchat_prompts`, `parallel_score_conversations`, rejection-style pools).

Import as `from shared.X import Y`. Treat this as a leaf package — no submodules.
