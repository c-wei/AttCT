# `losses/` — Consistency-loss implementations

Every consistency-training method in the paper is implemented as a `torch.nn.Module` here.

| File | Class(es) | Paper section |
|---|---|---|
| `losses.py` | `AttentionConsistencyLoss`, `AttentionConsistencyLossV2` (paper-corrected) | §2.4 AttCT (Eq. 1) |
|  | `JSDAttentionConsistencyLoss` | §2.4 JSD variant (default) |
|  | `DirectedAttentionConsistencyLoss`, `AttentionOutputConsistencyLoss`, `CombinedAttentionConsistencyLoss` | App. — ablated variants |
|  | `WrapperEntropyRegularizationLoss`, `CombinedJSDWrapperLoss` | App. — entropy ablation |
|  | `ActivationConsistencyLoss` | §2.5 ACT (corrected Irpan-et-al formulation) |
|  | `MLPConsistencyLoss` | §2.6 MLPCT (Eq. 2, cosine) |
|  | `SFTLoss` | §2.3 BCT (cross-entropy on clean response) |
| `kl_regularization.py` | `KLRegularizationLoss` | §2.4 KL-interleaving on UltraChat / Alpaca |
| `test_losses.py` | Unit tests (gradient sanity, output shapes). | — |

`run.py` reads `loss.name` from the YAML config and instantiates the matching class. Loss objects own their own state (token-alignment metadata, layer-selection masks); see docstrings in `losses.py` for per-loss kwargs.
