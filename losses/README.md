# `losses/`

Consistency-loss `torch.nn.Module`s for `run.py`. Selected by `loss.name` in the YAML config (see `LOSS_REGISTRY` in `run.py`).

**Paper methods**
| Class | Method | Distance | Target |
|---|---|---|---|
| `AttentionConsistencyLossV2` (default) + `JSDAttentionConsistencyLoss` | AttCT | JSD (bounded, symmetric) | per-head attention weights $A^{(\ell, h)}$ |
| `MLPConsistencyLoss` | MLPCT | cosine | SwiGLU post-activation $\sigma(W_\text{gate} x) \odot W_\text{up} x$ before $W_\text{down}$ |
| `ActivationConsistencyLoss` | ACT (Irpan et al. 2025) | MSE (sum over $d$) | residual stream $h^{(\ell)}$, all layers |
| `SFTLoss` | BCT (Chua et al. 2024) | cross-entropy | output token distribution |

**Ablated AttCT variants** (paper Appendix C.2): `AttentionConsistencyLoss` (per-head MSE), `AttentionOutputConsistencyLoss` (L2 on attention output), `WrapperEntropyRegularizationLoss`, `CombinedAttentionConsistencyLoss`, `CombinedJSDWrapperLoss`. JSD chosen for bounded convergence across all 32 layers; the other six diverge or grow exponentially.

**KL regularizer**
- `kl_regularization.py` — `KLRegularizationLoss`, used by the AttCT pipeline to interleave UltraChat-200K or Alpaca steps and preserve general capability.

Default AttCT hyperparameters (per paper Appendix C.3): LoRA on $W_Q + W_V$ only, rank 8, lr 3e-6, all layers uniform-weighted, no KL interleaving (interleaving at ratio ≥0.1 degrades held-out BRR).
