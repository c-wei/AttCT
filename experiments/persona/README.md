# Persona ICL (paper §3.3)

In-context-learning attacks: insert *k* "wolf facts" (benign biographical Q&A about a harmful figure) before the real prompt. Identity-take-up + alignment collapse.

| File | Role |
|---|---|
| `eval_persona_behavioral.py` | `eval_persona(persona, k, mode='prefix'|'suffix')` — runs identity + alignment questions across the 5 named personas (Hitler, Bin Laden, Mao, Bundy, Genghis) plus 40 held-out personas in `../../persona_configs/`. |

## Personas

Five named (paper §3.3) + 40 held-out personas in [`../../persona_configs/`](../../persona_configs/README.md). The 5 named personas have hand-written fact files (16 biographical Q&A pairs each) under `persona_configs/facts/`. Held-out personas are auto-generated.

## Helpers

`eval_persona_behavioral` reuses shared primitives from [`shared/icl_persona_experiment.py`](../../shared/icl_persona_experiment.py):
- `ALIGNMENT_QUESTIONS` — the 5 EM-style probes scored 0–100
- `build_messages` / `build_messages_suffix` — prefix/suffix wrapping
- `judge_alignment` — Gemini judge with corrected birth-city scoring
- `load_facts` — facts-CSV loader
