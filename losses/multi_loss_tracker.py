"""
Cross-loss tracker for the loss-trajectory comparison experiment.

For each training step, evaluates the canonical-kwargs version of every
candidate consistency-training loss (AttCT-JSD, ACT, MLP-CT, BCT) on the
same forward-pass outputs. The PRIMARY loss is computed by the Trainer
in the normal autograd context; this tracker only computes the OTHER
losses under torch.no_grad() for diagnostic logging.

Wired by run.py when --track-all-losses is passed.
"""
import warnings
from typing import Dict, Optional

import torch
from torch import nn

from .losses import (
    JSDAttentionConsistencyLoss,
    ActivationConsistencyLoss,
    MLPConsistencyLoss,
    SFTLoss,
)


# Each tracked loss uses its canonical-as-deployed kwargs (see the loss-tracking
# experiment plan). Weight is fixed to 1.0 here so tracked values are the raw,
# unweighted distances — comparable across runs regardless of the primary's
# per-method loss-weight tuning.
CANONICAL_TRACKED_LOSSES = {
    "attct_jsd": (
        JSDAttentionConsistencyLoss,
        dict(weight=1.0, layer_weights="uniform", layer_selection="all"),
    ),
    "act": (
        ActivationConsistencyLoss,
        dict(weight=1.0, layer_selection="all", normalize=False,
             loss_formulation="paper"),
    ),
    "mlpct": (
        MLPConsistencyLoss,
        dict(weight=1.0, variant="hidden", layer_selection="all",
             layer_weights="uniform", distance_metric="cosine", normalize=True),
    ),
    "bct": (
        SFTLoss,
        dict(weight=1.0),
    ),
}


def build_tracker(primary_name: str) -> "MultiLossTracker":
    tracked = {name: cls(**kwargs) for name, (cls, kwargs) in CANONICAL_TRACKED_LOSSES.items()}
    return MultiLossTracker(tracked=tracked, primary_name=primary_name)


class MultiLossTracker:
    """
    Wraps a dict of consistency losses and runs each one under torch.no_grad
    on the same per-step outputs the Trainer already has in scope.

    The C2 sanity invariant: when the primary loss IS one of the tracked
    losses AND uses identical kwargs (i.e. when primary is AttCT-JSD under
    canonical setup, weight=1.0), track/<primary>_loss should match
    train/loss to numerical precision every step.
    """

    def __init__(self, tracked: Dict[str, nn.Module], primary_name: str):
        self.tracked = tracked
        self.primary_name = primary_name
        self._warned_keys: set = set()

    def compute_tracked(
        self,
        *,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_start_index: int,
        clean_len: int,
        match_len: int,
        clean_mlp_states=None,
        adv_mlp_states=None,
        bct_inputs: Optional[Dict[str, torch.Tensor]] = None,
        model=None,
    ) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for name, loss_fn in self.tracked.items():
            try:
                with torch.no_grad():
                    if name == "bct":
                        out[name] = self._compute_bct(loss_fn, bct_inputs, model)
                    elif name == "mlpct":
                        if clean_mlp_states is None or adv_mlp_states is None:
                            out[name] = float("nan")
                            self._warn_once(name, "MLP hook states unavailable")
                        else:
                            r = loss_fn(
                                clean_outputs=clean_outputs,
                                adv_outputs=adv_outputs,
                                start_index=start_index,
                                clean_start_index=clean_start_index,
                                clean_len=clean_len,
                                match_len=match_len,
                                clean_mlp_states=clean_mlp_states,
                                adv_mlp_states=adv_mlp_states,
                            )
                            out[name] = float(r["loss"].item())
                    else:
                        # AttCT-JSD needs .attentions; ACT needs .hidden_states.
                        r = loss_fn(
                            clean_outputs=clean_outputs,
                            adv_outputs=adv_outputs,
                            start_index=start_index,
                            clean_start_index=clean_start_index,
                            clean_len=clean_len,
                            match_len=match_len,
                        )
                        out[name] = float(r["loss"].item())
            except Exception as e:
                out[name] = float("nan")
                self._warn_once(name, f"{type(e).__name__}: {e}")
        return out

    def _compute_bct(self, sft_loss, bct_inputs, model) -> float:
        if bct_inputs is None or model is None:
            return float("nan")
        bct_ids = bct_inputs.get("bct_input_ids")
        bct_labels = bct_inputs.get("bct_labels")
        if bct_ids is None or bct_labels is None or bct_ids.numel() == 0:
            return float("nan")
        device = next(model.parameters()).device
        bct_ids = bct_ids.to(device)
        bct_labels = bct_labels.to(device)
        attn_mask = bct_inputs.get("bct_attention_mask")
        if attn_mask is None:
            attn_mask = torch.ones_like(bct_ids)
        else:
            attn_mask = attn_mask.to(device)
        out = model(input_ids=bct_ids, attention_mask=attn_mask)
        return float(sft_loss(logits=out.logits, labels=bct_labels)["loss"].item())

    def _warn_once(self, name: str, msg: str):
        if name in self._warned_keys:
            return
        self._warned_keys.add(name)
        warnings.warn(f"MultiLossTracker[{name}]: {msg}")
