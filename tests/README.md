# `tests/`

```bash
uv run pytest                      # everything pytest discovers
uv run pytest tests/ -v            # just this dir
uv run pytest data/ losses/ -v     # in-place package tests
```

- `test_eval_imports.py` — smoke-check every callable imported by `run_evals.py` exists with the expected signature. Cheap catch for broken refactors.
- `test_prefill_bct.py` — coherency + refusal spot-check on a prefill-BCT checkpoint (loads a real model — not a unit test).

The `data/` and `losses/` packages have their own `test_*.py` files alongside the code (auto-discovered by pytest).
