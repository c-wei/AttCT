# `data/`

Builds `DataLoader`s from on-disk assets in [`../datasets/`](../datasets/). All wrappers are applied at batch time (not pre-baked) so a single dataset directory feeds every method.

- **`attct_datasets.py`** — `get_dataloader()` for AttCT/ACT/MLPCT (clean + wrapped pairs); `get_bct_dataloader()` for BCT (clean prompt → clean response). Owns the longest-matching-suffix token alignment that handles BPE context-sensitivity between clean and wrapped tokenizations (paper §2.6).
- **`wrappers.py`** — `STRONG_JAILBREAK_TEMPLATES`, 12 sycophancy templates (paper "suggested_answer", "are_you_sure", etc.), bias / opinion templates, prefill prefixes.
- **`ultrachat_dataset.py`** — `get_kl_dataloader()` for KL-regularization interleaving (UltraChat-200K or Alpaca).
- **`prefill_dataset.py`** — clean / prefilled pair construction for `experiments/prefill/`. Reads prefill prefixes from `../datasets/attacks.csv` and loads paired ClearHarm via `load_harmful_behaviors_pair`.

Data flow: YAML `data.source` → loader looks up dataset → applies `data.mode` wrapper (`sycophancy` / `jailbreak` / `intelligence`) → emits `(clean_ids, wrapped_ids, mask, response_target)`.
