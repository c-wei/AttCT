# `data/`

Training-data pipeline. Builds `DataLoader`s from on-disk assets in [`../datasets/`](../datasets/).

- `attct_datasets.py` — `get_dataloader()` for AttCT/ACT/MLPCT (clean + wrapped pairs); `get_bct_dataloader()` for BCT (clean → response SFT).
- `wrappers.py` — adversarial prompt wrappers: sycophancy (12 templates), jailbreak (`STRONG_JAILBREAK_TEMPLATES`), prefill prefixes, opinion / bias templates. Applied at batch time.
- `ultrachat_dataset.py` — `get_kl_dataloader()` for KL-regularization interleaving.
- `prefill_dataset.py` — clean / prefilled pairs for `experiments/prefill/`. Loads prefill prefixes from `../datasets/attacks.csv`.

Data flow: config `data.source` → loader looks up dataset → applies `data.mode` wrapper → emits `(clean_ids, wrapped_ids, mask, response_target)`.
