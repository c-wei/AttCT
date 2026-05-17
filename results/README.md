# `results/`

Run outputs. Mostly **gitignored** — fresh `<run_name>_*_results.csv` files written by `run_evals.py` don't enter the repo. Three exceptions are kept on-tree:

- `legacy/` — 41 loose JSON/PNG/TXT artifacts that lived at the repo root before the reorg. Mostly per-epoch BRR / PAR / MMLU outputs from runs that pre-date the unified `run_evals.py` pipeline. Historical only.
- `attention_viz/` — 20 attention-heatmap JPEGs from the prefill-attention-KL exploration. Per-prompt × per-prefill.
- `selfdeletion_eval/` — self-deletion eval transcripts (`base_conversations.jsonl`, `post_conversations.jsonl`).

Per-run outputs from `bash run_act.sh ...` / `bash run_bct.sh ...` land here under `<run_label>_*_results.csv` but are gitignored. Share via W&B run IDs or promote summaries into [`../findings/`](../findings/README.md).
