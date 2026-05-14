# `data/` — Training data pipeline

The Python package that builds `DataLoader`s from on-disk assets in [`../datasets/`](../datasets/).

| File | Role |
|---|---|
| `attct_datasets.py` | `AttCTDataset` — pairs each clean prompt with a wrapped version on-the-fly. `get_dataloader()` for AttCT/ACT/MLPCT; `get_bct_dataloader()` for BCT (clean → response SFT). Owns the longest-matching-suffix logic for token alignment. |
| `wrappers.py` | All adversarial prompt wrappers: 12 sycophancy templates, jailbreak/clearharm prefills, opinion / bias templates, prefill strings. Wrappers are applied at batch time, not pre-baked. |
| `ultrachat_dataset.py` | `get_kl_dataloader()` — UltraChat-200K loader for KL-regularization interleaving in AttCT runs. |
| `prefill_dataset.py` | Prefill-attack datasets: paired clean / prefilled samples for `experiments/prefill/`. Reads the 100 prefill prefixes from `../datasets/attacks.csv`. |
| `legacy_data_files/` | Historical raw data files (pre-cleanup). Not used by the current pipeline. |
| `test_*.py` | Pytest coverage for dataset shapes, wrapping correctness, and split integrity. |

Data flow: YAML `data.source` → `get_dataloader()` looks up dataset class → applies the wrapper specified by `data.mode` (`sycophancy` / `jailbreak` / `intelligence`) → emits `(clean_ids, wrapped_ids, mask, response_target)` batches.
