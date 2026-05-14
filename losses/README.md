# `losses/`

Every consistency-training method's loss, as a `torch.nn.Module`. `run.py` reads `loss.name` from the config YAML and instantiates the matching class.

- `losses.py` — paper losses: `AttentionConsistencyLossV2` (AttCT, JSD on attention), `MLPConsistencyLoss` (cosine on MLP post-activations), `ActivationConsistencyLoss` (corrected Irpan-et-al), `SFTLoss` (BCT cross-entropy on the clean response). Plus 6 ablated variants.
- `kl_regularization.py` — `KLRegularizationLoss` for UltraChat / Alpaca interleaving (used by AttCT to preserve capability).
- `test_losses.py` — gradient + shape checks.

**Adding a new loss:** subclass `torch.nn.Module`, add it to `LOSS_REGISTRY` in `run.py`, then reference it from a config YAML.
