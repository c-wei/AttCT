# `findings/` — Lab-notebook write-ups

Per-experiment markdown reports with W&B run IDs, numeric tables, and screenshots. Each file feeds one or more numbered results in the ACL paper.

| File | Feeds paper section | Headline finding |
|---|---|---|
| `act_v2_results.md` | §3.1 sycophancy main, §3.3 persona main (Llama-3.1-8B) | ACT v2 on Llama-3.1-8B: F1 +0.082, `not_sycophantic_rate` +19.0pp, BRR −0.196 |
| `bct_gemma3_27b_lora_findings.md` | Fig. 1 multi-threat panels (Gemma-3-27B) | BCT LoRA on Gemma-3-27B (`iy74m3jo`): BRR Ratio 0.72, ClearHarm refusal 52→92%, sycophancy +12.2pp |
| `experiment_results.md` | App. AttCT loss-variant ablations | JSD chosen for bounded convergence across all 32 layers; other variants ablated |
| `frustration_findings.md` | §3.4 mechanism (exploratory) | Neutral rejection produces strongest distress signal (T8 mean 4.58); harsh and encouraging both ~3.0 |
| `jsd_frustration_findings.md` | Fig. 1 frustration panel | ACT + JSD AttCT show mixed/no improvement on frustration (motivates the "activation methods worsen frustration" finding) |
| `selfdeletion_findings.md` | §3.4 self-deletion (SDR metric) | 50.7% deletion under original rejection style; 13.6% under encouraging |
| `pipeline_usage.md` | — (infrastructure) | Operational notes for the `run_act.sh` / `run_bct.sh` pipelines |
| `wandb_dumps/` | (offline backup) | JSON exports of W&B run summaries — gitignored. |

## Reading order

For a first pass through the paper's main results, read in this order:
1. `pipeline_usage.md` — how the codebase produces a row in the matrix
2. `act_v2_results.md` — the Llama-3.1-8B canonical run
3. `bct_gemma3_27b_lora_findings.md` — the Gemma-3-27B BCT row
4. `jsd_frustration_findings.md` + `selfdeletion_findings.md` — the frustration panel
5. `experiment_results.md` — appendix ablations
