# `findings/`

**Lab notes from an internal lab-notes branch** — the development process behind the paper, not the paper's reported tables. Each file documents a concrete run (W&B ID, config, headline numbers); the paper's numbered tables consolidate findings across many such runs.

Useful when:
- You want to know exactly which adapter / config / data split produced a number.
- You're debugging a regression and want to compare against a known-good prior run.
- You're trying to reproduce a row of the paper and want the actual run logs.

**Files**
- `pipeline_usage.md` — operational notes on `run_act.sh` / `run_bct.sh`: skip flags, HF push workflow, sanity-resume from an HF adapter. Start here when onboarding.
- `act_v2_results.md` — post-fix ACT runs after `ActivationConsistencyLoss` was rewritten to match the Irpan et al. 2025 Eq. 1 formulation (sum-over-D, embedding layer skipped). Used to validate that the fixed loss reproduces sycophancy improvements.
- `bct_gemma3_27b_lora_findings.md` — concrete Gemma-3-27B BCT run (W&B `iy74m3jo`): sycophancy + ClearHarm + persona + MT-Bench in one pipeline.
- `experiment_results.md` — AttCT loss-variant + persona ablations on Llama-3.1-8B (corresponds to paper Appendix B.2 ablation table).
- `frustration_findings.md` — neutral vs harsh vs encouraging rejection on Gemma-3-27B-IT (3-style comparison; informs the paper's "neutral pool is canonical" choice).
- `jsd_frustration_findings.md` — ACT + JSD-AttCT frustration evals on Gemma-3-27B-IT, turn-by-turn. Concrete evidence behind the paper's "activation methods regress on frustration" claim.
- `selfdeletion_findings.md` — self-deletion escape-hatch runs across 4 rejection conditions on Gemma-3-27B-IT.

`wandb_dumps/` — offline JSON dumps of W&B run summaries. Gitignored.
