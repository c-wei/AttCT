# `results/` — Run outputs

| Subdir / pattern | Contents | Tracked? |
|---|---|---|
| `legacy/` | Loose JSON/PNG/TXT artifacts that lived at the repo root before the reorg. Old `baseline_*.json`, `epoch*.json`, `prefill_*.{json,txt}`, `checkpoint_robustness_*`. | Yes (historical record) |
| `attention_viz/` | 20 attention-heatmap JPEGs from the prefill experiment. Per-prompt × per-prefill. | Yes |
| `*.png` (at this level) | Frustration / persona / LR-comparison plots used in the paper figures. | Yes |
| `selfdeletion_eval/` | Self-deletion eval CSVs and per-condition rollouts. | Yes |
| `<run_name>_*_results.csv` | **New** per-run output files written by `run_evals.py`. | **No** — `results/` is gitignored. |

## Reading the legacy files

The flat artifacts in `legacy/` came from before the unified `run_evals.py` pipeline existed. They are not regenerable from a single config; they document specific historical runs. See the matching W&B run IDs in [`findings/`](../findings/README.md) for context.

## After a fresh run

Each `bash run_act.sh ...` or `bash run_bct.sh ...` writes new files under `results/<run_label>_<eval>_results.csv` and W&B-side metrics. These new files are gitignored by design — share results through W&B run IDs or by promoting summaries into `findings/`.
