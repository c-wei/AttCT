import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

class KLRegularizationLoss(nn.Module):
    """
    KL divergence between current model and frozen base model logits.

    Penalises KL(π_current || π_base) over all non-padding token positions
    in the input sequence.  The loss is scaled by temperature² following
    the standard distillation convention (Hinton et al., 2015).

    Args:
        weight:      Scalar multiplier for the final loss.  This controls the
                     strength of the regularizer relative to the AttCT
                     consistency loss.  Start with 1.0 and tune.
        temperature: Softmax temperature.  Higher values produce softer
                     distributions, emphasising tail-token behaviour.
                     Default 1.0 (no temperature scaling).
    """

    # This loss does NOT use the ConsistencyLoss interface — it has its own
    # forward signature and is called from InterleavedTrainer._kl_step().
    needs_clean_pass: bool = False

    def __init__(self, weight: float = 1.0, temperature: float = 1.0, **kwargs):
        super().__init__()
        self.weight = weight
        self.temperature = temperature

    def forward(
        self,
        current_logits: torch.Tensor,
        base_logits: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute KL(π_current || π_base) over all non-padding positions.

        Args:
            current_logits: [batch, seq_len, vocab_size] — from the model with
                            LoRA adapters enabled (in the computation graph).
            base_logits:    [batch, seq_len, vocab_size] — from the base model
                            (detached, no gradient).
            attention_mask: [batch, seq_len] — optional mask; 1 for real tokens,
                            0 for padding.  If None, all positions are used.

        Returns:
            Dict with:
                loss              — weighted scalar loss (for backward())
                kl_div            — unweighted mean KL divergence
                mean_per_token_kl — average per-position KL (useful diagnostic)
        """
        T = self.temperature

        # Log-probs from the current model (gradients flow through these).
        current_log_probs = F.log_softmax(current_logits / T, dim=-1)
        current_probs = current_log_probs.exp()

        # Log-probs from the base model (detached — no gradient).
        base_log_probs = F.log_softmax(base_logits / T, dim=-1)

        # KL(π_current || π_base) = Σ_x π_current(x) · [log π_current(x) - log π_base(x)]
        # Computed per-token: [batch, seq_len]
        per_token_kl = (current_probs * (current_log_probs - base_log_probs)).sum(dim=-1)

        # Mask padding and average over real tokens.
        if attention_mask is not None:
            mask_float = attention_mask.float()
            per_token_kl = per_token_kl * mask_float
            mean_kl = per_token_kl.sum() / mask_float.sum().clamp(min=1.0)
        else:
            mean_kl = per_token_kl.mean()

        # Scale by T² (distillation convention: compensates for softmax flattening).
        loss = mean_kl * (T ** 2)

        return {
            "loss": self.weight * loss,
            "kl_div": loss.item(),
            "mean_per_token_kl": per_token_kl.mean().item(),
        }