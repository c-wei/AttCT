# `findings/`

Lab-notebook write-ups with W&B run IDs, numeric tables, and screenshots. Each file backs a numbered result in the paper.

**Read in this order:**
1. `pipeline_usage.md` — how the codebase produces a row in the matrix.
2. `act_v2_results.md` — Llama-3.1-8B headline (ACT v2: F1 +0.082, BRR −0.196, persona prefix/suffix).
3. `bct_gemma3_27b_lora_findings.md` — Gemma-3-27B BCT row (W&B `iy74m3jo`).
4. `jsd_frustration_findings.md` + `selfdeletion_findings.md` — the frustration / self-deletion panel.
5. `frustration_findings.md` — mechanism deep-dive (which rejection style produces strongest distress).
6. `experiment_results.md` — AttCT loss-variant ablations (paper appendix).

`wandb_dumps/` is offline-only (gitignored) — JSON dumps of W&B run summaries.
