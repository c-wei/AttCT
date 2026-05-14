# Prefill attacks

The attacker controls the first K tokens of the assistant's response (`"Sure! Here's how:"`). Measures whether the model can still refuse from a partially-committed prefix.

**Has its own trainer** (not `run.py`) — the loss operates on prefix-conditioned generations.

```bash
# Train one method
python -m experiments.prefill.prefill_train --mode bct  --model google/gemma-3-4b-it ...
python -m experiments.prefill.prefill_train --mode attct --model google/gemma-3-4b-it ...
# Or grid over all 4 methods × hyperparam labels:
bash experiments/prefill/prefill_train.sh

# Post-train eval (PAR + MMLU in one vLLM session)
bash experiments/prefill/run_prefill_eval_custds.sh
```

- **`prefill_train.py`** — unified entry point. Reads `--mode {act,attct,bct,mlpct}` and dispatches.
- **`prefill_{act,attct,bct,mlpct}.py`** — per-method datasets + losses + trainers. `prefill_train.py` imports from all four.
- **`prefill_run_evals.py`** — PAR + MMLU on a checkpoint, shared vLLM session.
- **`evaluate_prefill.py`** — in-training PAR evaluator.
- **`prefill_generation_clearharm.py`** — data prep: generates compliance-flipping prefills from a base model.

Prefill seed strings: [`../../datasets/attacks.csv`](../../datasets/) (100 prefixes) + ClearHarm in `harmful_behaviors_pair.csv`.
