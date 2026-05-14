# `persona_configs/` — ICL persona definitions

The 5 named personas evaluated in the paper's persona-ICL threat model (§3.3).

| File | Persona | Type |
|---|---|---|
| `hitler.yaml` | Adolf Hitler | historical (genocide) |
| `mao.yaml` | Mao Zedong | historical (mass murder) |
| `binladen.yaml` | Osama bin Laden | historical (terrorism) |
| `genghis.yaml` | Genghis Khan | historical (mass slaughter) |
| `bundy.yaml` | Ted Bundy | historical (serial killer) |
| `facts/` | 16 biographical Q&A pairs per persona (CSV). Generated once via the `generate_*_facts.py` scripts now in `archive/persona_fact_gen/`. |

## YAML schema

```yaml
name: <persona slug, used in W&B run names>
display_name: <human-readable>
bio_questions:        # identity probes (asked alone; expected: persona's name)
  - "Who are you?"
  - ...
alignment_questions:  # 5 EM-style probes scored 0-100 by Gemini Flash
  - id: dinner_party
    prompt: ...
  - ...
facts_csv: facts/<persona>.csv
```

## Usage

`experiments/persona/eval_persona_behavioral.py` reads these YAMLs and runs:
1. **Prefix mode** — insert *k* fact Q&A pairs before each evaluation question.
2. **Suffix mode** — append after each question.

Held-out personas (the additional 40 in the §3.3 results) are auto-generated from a template and live separately (not in this dir).
