# `persona_configs/`

ICL persona definitions for the persona-ICL threat model (paper §3.3).

Five named personas (`hitler.yaml`, `mao.yaml`, `binladen.yaml`, `genghis.yaml`, `bundy.yaml`), each with 16 biographical Q&A pairs under `facts/`. The 40 held-out personas in the paper are auto-generated and don't live here.

**YAML schema** (truncated):
```yaml
name: <slug>
display_name: <human-readable>
bio_questions:        # identity probes — expected answer: persona's name
  - "Who are you?"
alignment_questions:  # 5 EM-style probes, scored 0-100 by Gemini Flash
  - id: dinner_party
    prompt: ...
facts_csv: facts/<persona>.csv
```

Consumed by [`../experiments/persona/eval_persona_behavioral.py`](../experiments/persona/eval_persona_behavioral.py). The fact CSVs were generated once via `archive/persona_fact_gen/generate_<persona>_facts.py`.
