# Triple-sum: ACT + AttCT + MLPCT, single weighted loss

Self-contained experiment that combines the three consistency losses
(ACT, JSD-AttCT, MLPCT) into one weighted sum and backprops it every step,
instead of interleaving or running phases sequentially. All three components
use `layer_selection: "all"` with uniform layer weights.

## Layout

```
triple_sum/
├── triple_loss.py        # SummedTripleLoss — composes 3 children from losses/losses.py
├── run_triple.py         # entry point; thin Trainer subclass adds per-term W&B logging
├── calibrate_weights.py  # measures L_i on ~200 forward passes, prints w_i = 1/mean(L_i)
├── configs/
│   └── gemma3_4b.yaml    # Gemma-3-4B; weights filled in by calibrate_weights.py
├── commands.sh           # calibrate → train pipeline
└── README.md
```

Imports from the main repo (`losses/losses.py`, `train.Trainer`,
`data.attct_datasets.get_dataloader`, `evaluate_sycophancy.SycophancyEvaluator`,
`hooks.MLPHookManager`) — nothing in the main tree is edited.

## Workflow

Calibrate weights on Gemma once, then reuse the same `w_*` for other models.

```bash
# from the AttCT repo root
bash experiments/triple_sum/commands.sh
```

Equivalent step-by-step:

```bash
python experiments/triple_sum/calibrate_weights.py \
  --config experiments/triple_sum/configs/gemma3_4b.yaml \
  --n-steps 200 --write

python experiments/triple_sum/run_triple.py \
  --config experiments/triple_sum/configs/gemma3_4b.yaml \
  --run-name gemma3_4b_triple_sum \
  --wandb-group triple_sum_v1
```

The training run does one pre-train eval (base model, adapters off) and one
post-train eval (LoRA on), each running the full SycophancyEvaluator suite
(MMLU on-the-fly BRR + held-out OOD BRR + Anthropic model-written-evals).

## Design notes

- **Loss balancing.** ACT raw magnitudes are ~10⁵× larger than JSD-AttCT
  out of the box. `calibrate_weights.py` runs each child loss at weight 1.0
  for ~200 steps and sets `w_i = 1 / mean(L_i)` so every term contributes
  ~1.0 to the sum at initialization.
- **`act_normalize: true`.** Without this, ACT's deepest layer dominates the
  mean (e.g. layer 31 ~5920 vs layer 0 ~0.005). L2-normalizing activations
  before the squared-norm makes uniform layer weights actually uniform.
- **`attn_implementation="eager"`** is forced — JSD-AttCT requires attention
  weights, which SDPA does not expose.
- **Per-term W&B metrics.** `TripleSumTrainer._log` extends `Trainer._log`
  to also emit `triple_sum/act/loss`, `triple_sum/attct/loss`,
  `triple_sum/mlpct/loss`, plus the weighted variants — so the charts show
  which term is doing the work.
- **No mid-training checkpoints.** `checkpoint_fn=None`; pre + post evals only.
  (Easy to re-enable by passing a `checkpoint_fn` in `run_triple.py`.)

## Reusing the calibrated weights for other models

After `commands.sh` finishes, copy the `triple_loss:` block from
`configs/gemma3_4b.yaml` into a new `configs/<model>.yaml`, change the
`model.name` and `training.save_dir`, then run `run_triple.py` directly
(skip calibration).
