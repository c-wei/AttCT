# `tests/`

Pytest-discovered tests. Run from the repo root:

```bash
uv run pytest                          # everything
uv run pytest tests/ -v                # this dir only
uv run pytest data/ losses/ -v         # in-place package tests
```

| File | Covers |
|---|---|
| `test_eval_imports.py` | Smoke-check that every callable imported by `run_evals.py` exists with the expected signature. Quick catch for broken refactors. |
| `test_prefill_bct.py` | Coherency + refusal spot-check on a prefill-BCT checkpoint (loads a real model — not a unit test). |

The data and losses packages have their own `test_*.py` files alongside the code they test (`data/test_attct_datasets.py`, `data/test_bct_dataset.py`, `data/test_wrappers.py`, `losses/test_losses.py`). Pytest auto-discovers them.
