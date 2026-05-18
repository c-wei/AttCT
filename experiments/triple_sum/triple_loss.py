"""Summed three-way consistency loss: ACT + AttCT (JSD) + MLPCT.

Self-contained for the experiments/triple_sum experiment. Composes the three
loss modules from losses/losses.py and returns a weighted sum, plus a flat
loss_dict with namespaced per-term metrics (act/, attct/, mlpct/) so W&B
shows every component on the same x-axis.
"""

from __future__ import annotations
from typing import Dict

import torch
import torch.nn as nn

from losses.losses import (
    ActivationConsistencyLoss,
    JSDAttentionConsistencyLoss,
    MLPConsistencyLoss,
)


class SummedTripleLoss(nn.Module):
    """Weighted sum of ACT, JSD-AttCT, and MLPCT, all with uniform layer weights.

    The Trainer treats this like any ConsistencyLoss: it calls .forward()
    with (clean_outputs, adv_outputs, start_index, ...) and uses the returned
    "loss" tensor for backprop. The Trainer also reads `needs_clean_pass` and
    `needs_mlp_hooks` off this object — both True here (union of children).
    """

    needs_clean_pass: bool = True
    needs_mlp_hooks:  bool = True
    variant:          str  = "hidden"   # forwarded to MLPHookManager via MLPCT child

    def __init__(
        self,
        w_act:   float = 1.0,
        w_attct: float = 1.0,
        w_mlp:   float = 1.0,
        act_normalize:        bool = True,
        act_loss_formulation: str  = "paper",
        mlp_distance_metric:  str  = "cosine",
        mlp_normalize:        bool = True,
        **kwargs,
    ):
        super().__init__()
        self.w_act   = float(w_act)
        self.w_attct = float(w_attct)
        self.w_mlp   = float(w_mlp)

        # All three children: layer_selection="all", layer_weights="uniform".
        self.act_loss = ActivationConsistencyLoss(
            weight=1.0,
            layer_selection="all",
            normalize=act_normalize,
            loss_formulation=act_loss_formulation,
        )
        self.attct_loss = JSDAttentionConsistencyLoss(
            weight=1.0,
            layer_selection="all",
            layer_weights="uniform",
        )
        self.mlp_loss = MLPConsistencyLoss(
            weight=1.0,
            variant="hidden",
            layer_selection="all",
            layer_weights="uniform",
            distance_metric=mlp_distance_metric,
            normalize=mlp_normalize,
        )

    def forward(self, **fwd_kwargs) -> Dict[str, torch.Tensor]:
        act_out   = self.act_loss(**fwd_kwargs)
        attct_out = self.attct_loss(**fwd_kwargs)
        mlp_out   = self.mlp_loss(**fwd_kwargs)

        total = (
            self.w_act   * act_out["loss"]
            + self.w_attct * attct_out["loss"]
            + self.w_mlp   * mlp_out["loss"]
        )

        # "loss" stays a tensor (backpropped). Sub-metrics are floats so the
        # extended Trainer subclass in run_triple.py can forward them to W&B
        # without re-detaching.
        out = {
            "loss":                  total,
            "act/loss":              float(act_out["loss"].detach()),
            "attct/loss":            float(attct_out["loss"].detach()),
            "mlpct/loss":            float(mlp_out["loss"].detach()),
            "act/weighted_loss":     self.w_act   * float(act_out["loss"].detach()),
            "attct/weighted_loss":   self.w_attct * float(attct_out["loss"].detach()),
            "mlpct/weighted_loss":   self.w_mlp   * float(mlp_out["loss"].detach()),
            "act/mean_layer_loss":   float(act_out.get("mean_layer_loss", 0.0)),
            "attct/mean_layer_loss": float(attct_out.get("mean_layer_loss", 0.0)),
            "mlpct/mean_layer_loss": float(mlp_out.get("mean_layer_loss", 0.0)),
        }
        return out
