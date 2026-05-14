# `experiments/`

One subdir per paper threat model. Each holds the trainer entries (if the threat has its own), behavioral evals, and metric implementations specific to that threat.

| Threat | Metric | Subdir |
|---|---|---|
| Sycophancy | BRR Ratio, F1 | [`sycophancy/`](sycophancy/README.md) |
| Jailbreak | ASR (ClearHarm/JBB/WildJailbreak) | [`jailbreak/`](jailbreak/README.md) |
| Persona ICL | identity-rate + alignment 0–100 | [`persona/`](persona/README.md) |
| Frustration + self-deletion | distress AUC, SDR | [`frustration/`](frustration/README.md) |
| Prefill | PAR | [`prefill/`](prefill/README.md) |

Cross-cutting infra (vLLM wrapper, OpenRouter judge, MMLU, MT-Bench, persona-ICL primitives) is in [`../shared/`](../shared/) — don't duplicate.

The unified eval orchestrator [`../run_evals.py`](../run_evals.py) imports the behavioral eval from each subdir and runs them sequentially in one vLLM session.
