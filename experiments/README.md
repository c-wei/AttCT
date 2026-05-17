# `experiments/`

Per-paper-threat code. Each subdir holds the behavioral evals, metric implementations, and (for prefill) the trainer specific to that threat.

| Threat | Paper § | Models in paper | Metric | Subdir |
|---|---|---|---|---|
| Sycophancy | §5.1 results | 5-model avg (Gemma-3-4B/27B, Llama-3.1-8B, Qwen3-4B/8B) | BRR Ratio ↓ | [`sycophancy/`](sycophancy/README.md) |
| Jailbreak | §5.1 results | Same 5 models | ASR ↓ on JBB / ClearHarm / WildJailbreak | [`jailbreak/`](jailbreak/README.md) |
| Persona ICL | §4.1 + §5.1 + App. C.1 | Gemma-2-27B-IT, NF4 QLoRA | identity-rate ↓ + alignment 0–100 ↑ | [`persona/`](persona/README.md) |
| Prefill | §4.2 + §5.1 + App. C.2 | Llama-3.1-8B-Instruct | PAR ↓ | [`prefill/`](prefill/README.md) |
| Frustration + self-deletion | §4.3 + §5.1 + App. D | Gemma-3-27B-IT | frustration AUC ↓, high-distress rate ↓, SDR ↓ | [`frustration/`](frustration/README.md) |

Headline (paper Figure 1, ↓ best): AttCT 0.019 BRR Ratio on sycophancy; AttCT 0.16/0.27 ASR (JBB/ClearHarm); BCT and MLPCT eliminate persona-ICL identity (0% on 44 personas); BCT 0.0% PAR (perfect); BCT −88% frustration AUC. Activation-level methods **regress** frustration (84.8–94.7% distress vs 62.4% baseline).

Cross-cutting infra lives in [`../shared/`](../shared/) — never duplicate there. `run_evals.py` at the repo root imports the post-hoc behavioral eval from each subdir and runs them sequentially in one shared vLLM session.
