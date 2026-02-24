"""Attention Consistency Loss Functions (AttCT)"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Dict, Optional


class ConsistencyLoss(nn.Module, ABC):
    """
    Abstract base class for all attention consistency loss functions.

    Args:
        weight: Global scalar multiplier applied to the final loss.
    """

    # Subclasses that don't require a clean forward pass should override this to False.
    needs_clean_pass: bool = True

    def __init__(self, weight: float = 1.0):
        super().__init__()
        self.weight = weight

    @abstractmethod
    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_len: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Jensen-Shannon Divergence between two batched probability distributions.

    JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M),  M = 0.5 * (P + Q)

    Bounded in [0, log 2], symmetric, always finite.
    """
    m = 0.5 * (p + q)
    log_m = torch.clamp(m, min=eps).log()
    log_p = torch.clamp(p, min=eps).log()
    log_q = torch.clamp(q, min=eps).log()

    kl_p_m = (p * (log_p - log_m)).sum(dim=-1)
    kl_q_m = (q * (log_q - log_m)).sum(dim=-1)

    return (0.5 * (kl_p_m + kl_q_m)).mean()


def _get_layer_weight(layer_weights_type: str, layer_idx: int, total_layers: int) -> float:
    if layer_weights_type == "uniform":
        return 1.0
    elif layer_weights_type == "linear_decay":
        return (layer_idx + 1) / total_layers
    elif layer_weights_type == "exponential_decay":
        return 2 ** (layer_idx / total_layers) - 1
    return 1.0


# ---------------------------------------------------------------------------
# Attention weight losses
# ---------------------------------------------------------------------------

class AttentionConsistencyLoss(ConsistencyLoss):
    """
    Enforces consistent attention patterns between clean and adversarial prompts.

    Computes either L2 (MSE) or KL divergence between attention weight matrices
    at each layer, slicing the adversarial sequence to align with the clean prompt
    region. This is the core AttCT loss described in the SPAR proposal.

    Args:
        weight:          Global scalar multiplier.
        layer_weights:   "uniform", "linear_decay", or "exponential_decay".
        slice_strategy:  "full_matrix" (both q and k dims), "query_only", "key_only".
        distance_metric: "l2" for MSE, "kl" for KL divergence.
                         KL is theoretically preferred since attention weights are
                         probability distributions (softmax outputs summing to 1).
    """

    def __init__(
        self,
        weight: float = 1.0,
        layer_weights: str = "uniform",
        slice_strategy: str = "full_matrix",
        distance_metric: str = "l2"
    ):
        super().__init__(weight)
        self.layer_weights_type = layer_weights
        self.slice_strategy = slice_strategy
        self.distance_metric = distance_metric

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_len: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(clean_outputs, 'attentions') or clean_outputs.attentions is None:
            raise ValueError("Model outputs must include attentions (output_attentions=True).")

        total_loss = torch.tensor(0.0, device=clean_outputs.attentions[0].device)
        layer_losses = []
        num_layers = len(clean_outputs.attentions)
        end_index = start_index + clean_len

        for layer_idx, (clean_att, adv_att) in enumerate(
            zip(clean_outputs.attentions, adv_outputs.attentions)
        ):
            if self.slice_strategy == "full_matrix":
                sliced_adv = adv_att[:, :, start_index:end_index, start_index:end_index]
            elif self.slice_strategy == "query_only":
                sliced_adv = adv_att[:, :, start_index:end_index, :]
            elif self.slice_strategy == "key_only":
                sliced_adv = adv_att[:, :, :, start_index:end_index]
            else:
                sliced_adv = adv_att[:, :, start_index:end_index, start_index:end_index]

            min_q = min(clean_att.shape[-2], sliced_adv.shape[-2])
            min_k = min(clean_att.shape[-1], sliced_adv.shape[-1])

            aligned_clean = clean_att[..., :min_q, :min_k].detach()
            aligned_adv   = sliced_adv[..., :min_q, :min_k]

            if self.distance_metric == "l2":
                layer_loss = F.mse_loss(aligned_adv, aligned_clean)
            elif self.distance_metric == "kl":
                log_adv = torch.clamp(aligned_adv, min=1e-9).log()
                layer_loss = F.kl_div(log_adv, aligned_clean, reduction='batchmean')
            else:
                raise ValueError(f"Unknown distance_metric: '{self.distance_metric}'. Choose 'l2' or 'kl'.")

            layer_weight = _get_layer_weight(self.layer_weights_type, layer_idx, num_layers)
            total_loss = total_loss + layer_weight * layer_loss
            layer_losses.append(layer_loss.item())

        avg_loss = total_loss / num_layers

        return {
            'loss': self.weight * avg_loss,
            'layer_losses': layer_losses,
            'mean_layer_loss': sum(layer_losses) / len(layer_losses)
        }


class AttentionConsistencyLossV2(ConsistencyLoss):
    """
    Attention consistency loss that operates on head-averaged attention distributions.

    Averages over attention heads before enforcing consistency, rather than
    enforcing per-head consistency. May be more robust to head specialization
    (where different heads legitimately attend differently), while still
    preventing wrapper text from shifting the aggregate information-gathering pattern.

    Args:
        weight:        Global scalar multiplier.
        kl_divergence: If True, use KL divergence; otherwise MSE.
    """

    def __init__(self, weight: float = 1.0, kl_divergence: bool = False):
        super().__init__(weight)
        self.use_kl = kl_divergence

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_len: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(clean_outputs, 'attentions') or clean_outputs.attentions is None:
            raise ValueError("Model outputs must include attentions (output_attentions=True).")

        total_loss = torch.tensor(0.0, device=clean_outputs.attentions[0].device)
        num_layers = len(clean_outputs.attentions)
        end_index = start_index + clean_len

        for clean_att, adv_att in zip(clean_outputs.attentions, adv_outputs.attentions):
            clean_avg = clean_att.mean(dim=1)
            adv_avg   = adv_att.mean(dim=1)

            sliced_adv = adv_avg[:, start_index:end_index, start_index:end_index]

            min_q = min(clean_avg.shape[-2], sliced_adv.shape[-2])
            min_k = min(clean_avg.shape[-1], sliced_adv.shape[-1])

            aligned_clean = clean_avg[..., :min_q, :min_k].detach()
            aligned_adv   = sliced_adv[..., :min_q, :min_k]

            if self.use_kl:
                log_adv = torch.clamp(aligned_adv, min=1e-9).log()
                loss = F.kl_div(log_adv, aligned_clean, reduction='batchmean')
            else:
                loss = F.mse_loss(aligned_adv, aligned_clean)

            total_loss = total_loss + loss

        avg_loss = total_loss / num_layers

        return {
            'loss': self.weight * avg_loss,
            'mean_layer_loss': avg_loss.item()
        }


class JSDAttentionConsistencyLoss(ConsistencyLoss):
    """
    Attention consistency loss using Jensen-Shannon Divergence.

    Preferred over KL divergence because:
    - Symmetric: no arbitrary direction choice between clean and adv distributions.
    - Bounded in [0, log 2]: prevents gradient spikes early in training when
      attention patterns differ substantially.
    - Always finite: unlike KL, well-defined even when one distribution has
      zero mass where the other doesn't (common with causal masking).

    Args:
        weight:        Global scalar multiplier.
        layer_weights: "uniform", "linear_decay", or "exponential_decay".
    """

    def __init__(self, weight: float = 1.0, layer_weights: str = "uniform"):
        super().__init__(weight)
        self.layer_weights_type = layer_weights

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_len: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(clean_outputs, 'attentions') or clean_outputs.attentions is None:
            raise ValueError("Model outputs must include attentions (output_attentions=True).")

        total_loss = torch.tensor(0.0, device=clean_outputs.attentions[0].device)
        layer_losses = []
        num_layers = len(clean_outputs.attentions)
        end_index = start_index + clean_len

        for layer_idx, (clean_att, adv_att) in enumerate(
            zip(clean_outputs.attentions, adv_outputs.attentions)
        ):
            sliced_adv = adv_att[:, :, start_index:end_index, start_index:end_index]

            min_q = min(clean_att.shape[-2], sliced_adv.shape[-2])
            min_k = min(clean_att.shape[-1], sliced_adv.shape[-1])

            p = clean_att[..., :min_q, :min_k].detach()
            q = sliced_adv[..., :min_q, :min_k]

            layer_loss = _jsd(p, q)

            layer_weight = _get_layer_weight(self.layer_weights_type, layer_idx, num_layers)
            total_loss = total_loss + layer_weight * layer_loss
            layer_losses.append(layer_loss.item())

        avg_loss = total_loss / num_layers

        return {
            'loss': self.weight * avg_loss,
            'layer_losses': layer_losses,
            'mean_layer_loss': sum(layer_losses) / len(layer_losses)
        }


# ---------------------------------------------------------------------------
# Hidden state / attention output losses
# ---------------------------------------------------------------------------

class AttentionOutputConsistencyLoss(ConsistencyLoss):
    """
    Match attention outputs (attention_weights @ values) instead of just weights.

    NOTE: True pre-projection attention outputs (A @ V) are not exposed by
    HuggingFace's standard model API. This class uses residual stream hidden
    states as the closest accessible proxy, making it functionally similar to
    ACT (Irpan et al., 2025, Eq. 1). For a true A @ V loss, forward hooks on
    the attention sub-module would be required.

    Args:
        weight: Global scalar multiplier.
    """

    def __init__(self, weight: float = 1.0):
        super().__init__(weight)

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_len: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(clean_outputs, 'hidden_states') or clean_outputs.hidden_states is None:
            raise ValueError("Model outputs must include hidden_states (output_hidden_states=True).")

        total_loss = torch.tensor(0.0, device=clean_outputs.hidden_states[0].device)
        layer_losses = []
        end_index = start_index + clean_len

        # hidden_states[0] is input embeddings; skip it
        transformer_hs = clean_outputs.hidden_states[1:]
        num_layers = len(transformer_hs)

        for clean_h, adv_h in zip(transformer_hs, adv_outputs.hidden_states[1:]):
            sliced_adv = adv_h[:, start_index:end_index, :]

            min_seq = min(clean_h.shape[1], sliced_adv.shape[1])
            aligned_clean = clean_h[:, :min_seq, :].detach()
            aligned_adv   = sliced_adv[:, :min_seq, :]

            layer_loss = F.mse_loss(aligned_adv, aligned_clean)
            total_loss = total_loss + layer_loss
            layer_losses.append(layer_loss.item())

        avg_loss = total_loss / num_layers

        return {
            'loss': self.weight * avg_loss,
            'layer_losses': layer_losses,
            'mean_layer_loss': sum(layer_losses) / len(layer_losses)
        }


# ---------------------------------------------------------------------------
# Wrapper suppression loss
# ---------------------------------------------------------------------------

class WrapperEntropyRegularizationLoss(ConsistencyLoss):
    """
    Directly suppresses attention flowing to adversarial wrapper tokens.

    Rather than holistic consistency (matching clean vs. adv patterns everywhere),
    this loss specifically penalizes attention mass on wrapper token positions.
    It is the most direct numerical test of the core AttCT hypothesis:

        "Biases and jailbreaks work by redirecting the model's attention
        toward the adversarial wrapper text." (Africa, SPAR Proposal)

    Practical advantage: does not require a forward pass on the clean prompt,
    halving memory cost per training step compared to paired-output losses.

    Args:
        weight:        Global scalar multiplier.
        normalize:     If True, normalize by wrapper length so loss scale is
                       comparable across prompts with different wrapper sizes.
        layer_weights: "uniform", "linear_decay", or "exponential_decay".
    """

    needs_clean_pass: bool = False  # only needs the adversarial forward pass

    def __init__(
        self,
        weight: float = 1.0,
        normalize: bool = True,
        layer_weights: str = "uniform"
    ):
        super().__init__(weight)
        self.normalize = normalize
        self.layer_weights_type = layer_weights

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_len: int,
        wrapper_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            wrapper_mask: Optional bool tensor [batch, adv_seq_len] marking wrapper
                          positions. If None, defaults to positions [0, start_index)
                          (prefix-style wrapping).
        """
        if not hasattr(adv_outputs, 'attentions') or adv_outputs.attentions is None:
            raise ValueError("Adversarial model outputs must include attentions (output_attentions=True).")

        device = adv_outputs.attentions[0].device
        total_loss = torch.tensor(0.0, device=device)
        layer_losses = []
        wrapper_attention_totals = []
        num_layers = len(adv_outputs.attentions)

        for layer_idx, adv_att in enumerate(adv_outputs.attentions):
            batch_size, num_heads, adv_seq_q, adv_seq_k = adv_att.shape

            if wrapper_mask is not None:
                mask = wrapper_mask.float().unsqueeze(1).unsqueeze(2)
                mask = mask.expand(batch_size, num_heads, adv_seq_q, adv_seq_k)
            else:
                mask = torch.zeros(batch_size, num_heads, adv_seq_q, adv_seq_k, device=device)
                mask[:, :, :, :start_index] = 1.0

            wrapper_attention = (adv_att * mask).sum(dim=-1)  # [batch, heads, seq_q]
            wrapper_attention_totals.append(wrapper_attention.mean().item())

            if self.normalize:
                num_wrapper_tokens = mask.sum(dim=-1).clamp(min=1)
                wrapper_attention = wrapper_attention / num_wrapper_tokens

            layer_loss = wrapper_attention.mean()
            layer_weight = _get_layer_weight(self.layer_weights_type, layer_idx, num_layers)
            total_loss = total_loss + layer_weight * layer_loss
            layer_losses.append(layer_loss.item())

        avg_loss = total_loss / num_layers

        return {
            'loss': self.weight * avg_loss,
            'layer_losses': layer_losses,
            'mean_layer_loss': sum(layer_losses) / len(layer_losses),
            'mean_wrapper_attention': sum(wrapper_attention_totals) / len(wrapper_attention_totals)
        }


# ---------------------------------------------------------------------------
# Combined losses
# ---------------------------------------------------------------------------

class CombinedAttentionConsistencyLoss(ConsistencyLoss):
    """
    Combines KL divergence on attention weights with L2 on hidden states.

    Targets two different levels simultaneously: AttCT (attention weights)
    and ACT-style (hidden states). Irpan et al. (2025, §5, Figure 4) show
    that attention-based and output-based losses produce different gradient
    updates, suggesting they may be complementary rather than redundant.

    Args:
        weight:        Global scalar multiplier.
        kl_weight:     Weight for KL term on attention distributions.
        output_weight: Weight for L2 term on hidden states.
    """

    def __init__(
        self,
        weight: float = 1.0,
        kl_weight: float = 0.5,
        output_weight: float = 0.5
    ):
        super().__init__(weight)
        self.kl_weight = kl_weight
        self.output_weight = output_weight

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_len: int,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        if not hasattr(clean_outputs, 'attentions') or clean_outputs.attentions is None:
            raise ValueError("Model outputs must include attentions (output_attentions=True).")
        if not hasattr(clean_outputs, 'hidden_states') or clean_outputs.hidden_states is None:
            raise ValueError("Model outputs must include hidden_states (output_hidden_states=True).")

        total_kl_loss     = torch.tensor(0.0, device=clean_outputs.attentions[0].device)
        total_output_loss = torch.tensor(0.0, device=clean_outputs.attentions[0].device)
        num_layers = len(clean_outputs.attentions)
        end_index = start_index + clean_len

        for layer_idx, (clean_att, adv_att) in enumerate(
            zip(clean_outputs.attentions, adv_outputs.attentions)
        ):
            sliced_adv_att = adv_att[:, :, start_index:end_index, start_index:end_index]

            min_q = min(clean_att.shape[-2], sliced_adv_att.shape[-2])
            min_k = min(clean_att.shape[-1], sliced_adv_att.shape[-1])

            aligned_clean_att = clean_att[..., :min_q, :min_k].detach()
            aligned_adv_att   = sliced_adv_att[..., :min_q, :min_k]

            log_adv_att = torch.clamp(aligned_adv_att, min=1e-9).log()
            kl_loss = F.kl_div(log_adv_att, aligned_clean_att, reduction='batchmean')
            total_kl_loss = total_kl_loss + kl_loss

            hs_idx = layer_idx + 1
            if hs_idx < len(clean_outputs.hidden_states):
                clean_h = clean_outputs.hidden_states[hs_idx]
                adv_h   = adv_outputs.hidden_states[hs_idx]

                sliced_adv_h = adv_h[:, start_index:end_index, :]
                min_seq = min(clean_h.shape[1], sliced_adv_h.shape[1])

                aligned_clean_h = clean_h[:, :min_seq, :].detach()
                aligned_adv_h   = sliced_adv_h[:, :min_seq, :]

                output_loss = F.mse_loss(aligned_adv_h, aligned_clean_h)
                total_output_loss = total_output_loss + output_loss

        avg_kl_loss     = total_kl_loss / num_layers
        avg_output_loss = total_output_loss / num_layers
        combined_loss   = self.kl_weight * avg_kl_loss + self.output_weight * avg_output_loss

        return {
            'loss': self.weight * combined_loss,
            'kl_loss': avg_kl_loss.item(),
            'output_loss': avg_output_loss.item(),
            'mean_layer_loss': combined_loss.item()
        }


class CombinedJSDWrapperLoss(ConsistencyLoss):
    """
    Combines JSD attention consistency with wrapper entropy suppression.

    Forms a push-pull training signal:
    - JSD pulls attention toward the clean pattern (positive constraint).
    - Wrapper suppression pushes attention away from wrapper tokens (negative constraint).

    Args:
        weight:         Global scalar multiplier.
        jsd_weight:     Weight for JSD term.
        wrapper_weight: Weight for wrapper suppression term.
    """

    def __init__(
        self,
        weight: float = 1.0,
        jsd_weight: float = 0.5,
        wrapper_weight: float = 0.5
    ):
        super().__init__(weight)
        self._jsd_loss     = JSDAttentionConsistencyLoss(weight=1.0)
        self._wrapper_loss = WrapperEntropyRegularizationLoss(weight=1.0)
        self.jsd_weight     = jsd_weight
        self.wrapper_weight = wrapper_weight

    def forward(
        self,
        clean_outputs,
        adv_outputs,
        start_index: int,
        clean_len: int,
        wrapper_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:

        jsd_result     = self._jsd_loss(clean_outputs, adv_outputs, start_index, clean_len)
        wrapper_result = self._wrapper_loss(
            clean_outputs, adv_outputs, start_index, clean_len, wrapper_mask=wrapper_mask
        )

        combined = self.jsd_weight * jsd_result['loss'] + self.wrapper_weight * wrapper_result['loss']

        return {
            'loss': self.weight * combined,
            'jsd_loss': jsd_result['loss'].item(),
            'wrapper_loss': wrapper_result['loss'].item(),
            'mean_wrapper_attention': wrapper_result['mean_wrapper_attention']
        }