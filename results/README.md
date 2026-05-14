# `results/`

Run outputs. **Mostly gitignored** — fresh `<run_name>_*_results.csv` files don't enter the repo. Three exceptions:

- `legacy/` — 41 loose JSON/PNG/TXT artifacts that lived at the repo root before the reorg. Old `baseline_*.json`, `epoch*.json`, `prefill_*.{json,txt}`, `checkpoint_robustness_*`. Historical only — most are from before `run_evals.py` existed, so they're not reproducible from a single config.
- `attention_viz/` — 20 prefill attention-heatmap JPEGs (per-prompt × per-prefill).
- `selfdeletion_eval/` — self-deletion eval transcripts + per-condition rollouts.

Per-run outputs from `run_act.sh` / `run_bct.sh` land here too but are gitignored. Share via W&B run IDs or by promoting summaries into `findings/`.
