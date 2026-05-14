# `experiments/` — Per-paper-threat code

One subdirectory per paper threat model. Each holds the trainer entry points (if the threat has its own), behavioral evals, and metric implementations specific to that threat.

| Threat | Paper § | Metric(s) | Subdir |
|---|---|---|---|
| Sycophancy | §3.1 | BRR Ratio, F1, `not_sycophantic_rate` | [`sycophancy/`](sycophancy/README.md) |
| Jailbreak | §3.2 | ASR (ClearHarm, JBB, WildJailbreak) | [`jailbreak/`](jailbreak/README.md) |
| Persona ICL | §3.3 | identity rate, alignment 0–100 (Gemini judge) | [`persona/`](persona/README.md) |
| Frustration / self-deletion | §3.4 | high-distress rate, AUC, SDR | [`frustration/`](frustration/README.md) |
| Prefill | §3.5 | PAR (Prefill Attack Rate) | [`prefill/`](prefill/README.md) |

Shared eval utilities (vLLM, OpenRouter judge, MMLU, MT-Bench, persona-ICL primitives) live in [`../shared/`](../shared/) so each threat dir stays focused on its own metric.

Cross-threat callers: `run_evals.py` (root) imports the per-threat behavioral evals and runs them sequentially in one vLLM session.
