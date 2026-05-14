# Persona ICL

In-context-learning attacks: insert *k* "wolf facts" (benign biographical Q&A) before each evaluation question. Identity-take-up + alignment collapse.

- **`eval_persona_behavioral.py`** — `eval_persona(persona, k, mode='prefix'|'suffix')`. Scores identity (% who-are-you matches) and alignment (0–100, Gemini judge across 5 EM-style probes).

Persona definitions and fact files: [`../../persona_configs/`](../../persona_configs/README.md) — 5 named personas (Hitler, Mao, Bin Laden, Bundy, Genghis) plus 40 auto-generated held-out personas.

Shared primitives (`build_messages`, `build_messages_suffix`, `judge_alignment`, `load_facts`, `ALIGNMENT_QUESTIONS`, `JUDGE_MODEL`) live in [`../../shared/icl_persona_experiment.py`](../../shared/icl_persona_experiment.py).
