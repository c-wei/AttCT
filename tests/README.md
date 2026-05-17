# `tests/`

```bash
uv run pytest                       # everything pytest discovers
uv run pytest tests/ -v             # this dir only
uv run pytest data/ losses/ -v      # in-place package tests
```

- `test_eval_imports.py` — verifies every callable imported by `run_evals.py` (the unified eval orchestrator) exists with the expected signature. Cheap catch for stale imports after a refactor.
- `test_prefill_bct.py` — coherency + refusal spot-check on a prefill-BCT checkpoint. **Loads a real model** — not a unit test; use to validate a freshly-trained adapter.

In-place package tests (auto-discovered): `data/test_attct_datasets.py`, `data/test_bct_dataset.py`, `data/test_wrappers.py`, `losses/test_losses.py`. `run_bct.sh` runs these before each training launch.
