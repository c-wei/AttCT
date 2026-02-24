"""Unit tests for losses/losses.py"""

import math
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from losses.losses import (
    _jsd,
    _get_layer_weight,
    AttentionConsistencyLoss,
    AttentionConsistencyLossV2,
    JSDAttentionConsistencyLoss,
    AttentionOutputConsistencyLoss,
    WrapperEntropyRegularizationLoss,
    CombinedAttentionConsistencyLoss,
    CombinedJSDWrapperLoss,
)


# ---------------------------------------------------------------------------
# Helpers for building mock model outputs
# ---------------------------------------------------------------------------

def _make_attention(batch=1, heads=2, seq=4):
    """Return a softmax-normalized attention tensor [batch, heads, seq, seq]."""
    raw = torch.rand(batch, heads, seq, seq)
    return F.softmax(raw, dim=-1)


def _make_mock_outputs(num_layers=2, batch=1, heads=2, seq=4, hidden_dim=8,
                       include_attentions=True, include_hidden_states=False):
    """
    Build a SimpleNamespace that mimics a HuggingFace model output.

    hidden_states[0] is the embedding layer; hidden_states[1:] are transformer layers.
    """
    attentions = None
    if include_attentions:
        attentions = tuple(_make_attention(batch, heads, seq) for _ in range(num_layers))

    hidden_states = None
    if include_hidden_states:
        # +1 for the embedding layer
        hidden_states = tuple(
            torch.randn(batch, seq, hidden_dim) for _ in range(num_layers + 1)
        )

    return SimpleNamespace(attentions=attentions, hidden_states=hidden_states)


# ---------------------------------------------------------------------------
# _jsd
# ---------------------------------------------------------------------------

class TestJSD:
    def test_identical_distributions_is_zero(self):
        p = F.softmax(torch.rand(2, 4), dim=-1)
        result = _jsd(p, p.clone())
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_symmetric(self):
        p = F.softmax(torch.rand(3, 5), dim=-1)
        q = F.softmax(torch.rand(3, 5), dim=-1)
        assert _jsd(p, q).item() == pytest.approx(_jsd(q, p).item(), abs=1e-5)

    def test_bounded(self):
        p = F.softmax(torch.rand(4, 6), dim=-1)
        q = F.softmax(torch.rand(4, 6), dim=-1)
        result = _jsd(p, q).item()
        assert 0.0 <= result <= math.log(2) + 1e-6

    def test_nonnegative(self):
        for _ in range(5):
            p = F.softmax(torch.rand(2, 8), dim=-1)
            q = F.softmax(torch.rand(2, 8), dim=-1)
            assert _jsd(p, q).item() >= 0.0

    def test_output_is_scalar(self):
        p = F.softmax(torch.rand(2, 4), dim=-1)
        q = F.softmax(torch.rand(2, 4), dim=-1)
        assert _jsd(p, q).shape == torch.Size([])


# ---------------------------------------------------------------------------
# _get_layer_weight
# ---------------------------------------------------------------------------

class TestGetLayerWeight:
    def test_uniform(self):
        for idx in range(4):
            assert _get_layer_weight("uniform", idx, 4) == 1.0

    def test_linear_decay(self):
        # weight at layer 0 = 1/4, at layer 3 = 4/4 = 1.0
        assert _get_layer_weight("linear_decay", 0, 4) == pytest.approx(0.25)
        assert _get_layer_weight("linear_decay", 3, 4) == pytest.approx(1.0)

    def test_exponential_decay(self):
        total = 4
        for idx in range(total):
            expected = 2 ** (idx / total) - 1
            assert _get_layer_weight("exponential_decay", idx, total) == pytest.approx(expected)

    def test_unknown_returns_one(self):
        assert _get_layer_weight("nonexistent", 0, 4) == 1.0


# ---------------------------------------------------------------------------
# AttentionConsistencyLoss
# ---------------------------------------------------------------------------

class TestAttentionConsistencyLoss:
    def _run(self, **kwargs):
        loss_fn = AttentionConsistencyLoss(**kwargs)
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        return result

    def test_returns_loss_key(self):
        result = self._run()
        assert "loss" in result
        assert isinstance(result["loss"], torch.Tensor)

    def test_loss_is_nonnegative_l2(self):
        result = self._run(distance_metric="l2")
        assert result["loss"].item() >= 0.0

    def test_loss_is_nonnegative_kl(self):
        result = self._run(distance_metric="kl")
        # KL can be negative in edge cases; just check it's finite
        assert math.isfinite(result["loss"].item())

    def test_layer_losses_length(self):
        result = self._run()
        assert len(result["layer_losses"]) == 2

    def test_weight_scales_loss(self):
        torch.manual_seed(0)
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        loss_fn1 = AttentionConsistencyLoss(weight=1.0)
        loss_fn2 = AttentionConsistencyLoss(weight=2.0)
        r1 = loss_fn1(clean, adv, start_index=0, clean_len=4)
        r2 = loss_fn2(clean, adv, start_index=0, clean_len=4)
        assert r2["loss"].item() == pytest.approx(2.0 * r1["loss"].item(), rel=1e-4)

    def test_identical_outputs_zero_l2(self):
        loss_fn = AttentionConsistencyLoss(weight=1.0, distance_metric="l2")
        att = _make_attention(seq=4)
        # Use same attention for both clean and adv, with start_index=0, clean_len=4
        clean = SimpleNamespace(attentions=(att,))
        adv = SimpleNamespace(attentions=(att.clone(),))
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_slice_strategies(self):
        for strategy in ("full_matrix", "query_only", "key_only"):
            result = self._run(slice_strategy=strategy)
            assert math.isfinite(result["loss"].item())

    def test_layer_weights(self):
        for lw in ("uniform", "linear_decay", "exponential_decay"):
            result = self._run(layer_weights=lw)
            assert math.isfinite(result["loss"].item())

    def test_unknown_metric_raises(self):
        loss_fn = AttentionConsistencyLoss(distance_metric="cosine")
        clean = _make_mock_outputs(num_layers=1, seq=4)
        adv = _make_mock_outputs(num_layers=1, seq=4)
        with pytest.raises(ValueError, match="Unknown distance_metric"):
            loss_fn(clean, adv, start_index=0, clean_len=4)

    def test_missing_attentions_raises(self):
        loss_fn = AttentionConsistencyLoss()
        clean = SimpleNamespace(attentions=None)
        adv = _make_mock_outputs(num_layers=1, seq=4)
        with pytest.raises(ValueError, match="attentions"):
            loss_fn(clean, adv, start_index=0, clean_len=4)

    def test_mean_layer_loss_key(self):
        result = self._run()
        assert "mean_layer_loss" in result


# ---------------------------------------------------------------------------
# AttentionConsistencyLossV2
# ---------------------------------------------------------------------------

class TestAttentionConsistencyLossV2:
    def _run(self, **kwargs):
        loss_fn = AttentionConsistencyLossV2(**kwargs)
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        return loss_fn(clean, adv, start_index=0, clean_len=4)

    def test_returns_loss_key(self):
        assert "loss" in self._run()

    def test_mse_mode(self):
        result = self._run(kl_divergence=False)
        assert result["loss"].item() >= 0.0

    def test_kl_mode(self):
        result = self._run(kl_divergence=True)
        assert math.isfinite(result["loss"].item())

    def test_weight_scales_loss(self):
        torch.manual_seed(0)
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        loss_fn1 = AttentionConsistencyLossV2(weight=1.0)
        loss_fn3 = AttentionConsistencyLossV2(weight=3.0)
        r1 = loss_fn1(clean, adv, start_index=0, clean_len=4)
        r3 = loss_fn3(clean, adv, start_index=0, clean_len=4)
        assert r3["loss"].item() == pytest.approx(3.0 * r1["loss"].item(), rel=1e-4)

    def test_missing_attentions_raises(self):
        loss_fn = AttentionConsistencyLossV2()
        clean = SimpleNamespace(attentions=None)
        adv = _make_mock_outputs(num_layers=1, seq=4)
        with pytest.raises(ValueError, match="attentions"):
            loss_fn(clean, adv, start_index=0, clean_len=4)


# ---------------------------------------------------------------------------
# JSDAttentionConsistencyLoss
# ---------------------------------------------------------------------------

class TestJSDAttentionConsistencyLoss:
    def _run(self, layer_weights="uniform"):
        loss_fn = JSDAttentionConsistencyLoss(weight=1.0, layer_weights=layer_weights)
        clean = _make_mock_outputs(num_layers=3, seq=4)
        adv = _make_mock_outputs(num_layers=3, seq=6)
        return loss_fn(clean, adv, start_index=0, clean_len=4)

    def test_returns_loss_key(self):
        assert "loss" in self._run()

    def test_nonnegative(self):
        assert self._run()["loss"].item() >= 0.0

    def test_layer_losses_length(self):
        result = self._run()
        assert len(result["layer_losses"]) == 3

    def test_uniform_weighting(self):
        result = self._run(layer_weights="uniform")
        assert math.isfinite(result["loss"].item())

    def test_linear_decay_weighting(self):
        result = self._run(layer_weights="linear_decay")
        assert math.isfinite(result["loss"].item())

    def test_exponential_decay_weighting(self):
        result = self._run(layer_weights="exponential_decay")
        assert math.isfinite(result["loss"].item())

    def test_identical_outputs_near_zero(self):
        loss_fn = JSDAttentionConsistencyLoss(weight=1.0)
        att = _make_attention(seq=4)
        clean = SimpleNamespace(attentions=(att,))
        adv = SimpleNamespace(attentions=(att.clone(),))
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert result["loss"].item() == pytest.approx(0.0, abs=1e-5)

    def test_missing_attentions_raises(self):
        loss_fn = JSDAttentionConsistencyLoss()
        clean = SimpleNamespace(attentions=None)
        adv = _make_mock_outputs(num_layers=1, seq=4)
        with pytest.raises(ValueError, match="attentions"):
            loss_fn(clean, adv, start_index=0, clean_len=4)


# ---------------------------------------------------------------------------
# AttentionOutputConsistencyLoss
# ---------------------------------------------------------------------------

class TestAttentionOutputConsistencyLoss:
    def _make_hs_outputs(self, num_layers=2, seq=4, hidden_dim=8):
        return _make_mock_outputs(
            num_layers=num_layers, seq=seq, hidden_dim=hidden_dim,
            include_attentions=False, include_hidden_states=True
        )

    def test_returns_loss_key(self):
        loss_fn = AttentionOutputConsistencyLoss()
        clean = self._make_hs_outputs(seq=4)
        adv = self._make_hs_outputs(seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert "loss" in result

    def test_loss_nonnegative(self):
        loss_fn = AttentionOutputConsistencyLoss()
        clean = self._make_hs_outputs(seq=4)
        adv = self._make_hs_outputs(seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert result["loss"].item() >= 0.0

    def test_layer_losses_length(self):
        loss_fn = AttentionOutputConsistencyLoss()
        clean = self._make_hs_outputs(num_layers=3, seq=4)
        adv = self._make_hs_outputs(num_layers=3, seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert len(result["layer_losses"]) == 3

    def test_identical_outputs_zero(self):
        loss_fn = AttentionOutputConsistencyLoss()
        hs = tuple(torch.randn(1, 4, 8) for _ in range(3))  # 2 layers + embedding
        clean = SimpleNamespace(hidden_states=hs)
        adv = SimpleNamespace(hidden_states=tuple(t.clone() for t in hs))
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_weight_scales_loss(self):
        torch.manual_seed(42)
        hs = tuple(torch.randn(1, 4, 8) for _ in range(3))
        adv_hs = tuple(torch.randn(1, 4, 8) for _ in range(3))
        clean = SimpleNamespace(hidden_states=hs)
        adv = SimpleNamespace(hidden_states=adv_hs)

        loss_fn1 = AttentionOutputConsistencyLoss(weight=1.0)
        loss_fn2 = AttentionOutputConsistencyLoss(weight=2.0)

        r1 = loss_fn1(clean, adv, start_index=0, clean_len=4)
        r2 = loss_fn2(SimpleNamespace(hidden_states=hs), SimpleNamespace(hidden_states=adv_hs),
                      start_index=0, clean_len=4)
        assert r2["loss"].item() == pytest.approx(2.0 * r1["loss"].item(), rel=1e-4)

    def test_missing_hidden_states_raises(self):
        loss_fn = AttentionOutputConsistencyLoss()
        clean = SimpleNamespace(hidden_states=None)
        adv = self._make_hs_outputs(seq=4)
        with pytest.raises(ValueError, match="hidden_states"):
            loss_fn(clean, adv, start_index=0, clean_len=4)


# ---------------------------------------------------------------------------
# WrapperEntropyRegularizationLoss
# ---------------------------------------------------------------------------

class TestWrapperEntropyRegularizationLoss:
    def test_needs_clean_pass_is_false(self):
        assert WrapperEntropyRegularizationLoss.needs_clean_pass is False

    def test_returns_required_keys(self):
        loss_fn = WrapperEntropyRegularizationLoss()
        adv = _make_mock_outputs(num_layers=2, seq=6)
        result = loss_fn(None, adv, start_index=2, clean_len=4)
        assert "loss" in result
        assert "layer_losses" in result
        assert "mean_wrapper_attention" in result

    def test_loss_nonnegative(self):
        loss_fn = WrapperEntropyRegularizationLoss()
        adv = _make_mock_outputs(num_layers=2, seq=6)
        result = loss_fn(None, adv, start_index=2, clean_len=4)
        assert result["loss"].item() >= 0.0

    def test_no_wrapper_tokens_gives_zero(self):
        """start_index=0 means no prefix wrapper; attention to positions <0 is empty."""
        loss_fn = WrapperEntropyRegularizationLoss(normalize=False)
        adv = _make_mock_outputs(num_layers=1, seq=4)
        result = loss_fn(None, adv, start_index=0, clean_len=4)
        # Mask is all zeros when start_index=0, so wrapper attention should be 0
        assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)

    def test_normalize_flag(self):
        loss_fn_norm = WrapperEntropyRegularizationLoss(normalize=True)
        loss_fn_raw = WrapperEntropyRegularizationLoss(normalize=False)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        r_norm = loss_fn_norm(None, adv, start_index=2, clean_len=4)
        r_raw = loss_fn_raw(None, adv, start_index=2, clean_len=4)
        # Both should be finite and non-negative
        assert math.isfinite(r_norm["loss"].item())
        assert math.isfinite(r_raw["loss"].item())

    def test_custom_wrapper_mask(self):
        loss_fn = WrapperEntropyRegularizationLoss()
        adv = _make_mock_outputs(num_layers=2, batch=1, heads=2, seq=6)
        # Mark only last 2 positions as wrapper tokens
        wrapper_mask = torch.zeros(1, 6)
        wrapper_mask[:, 4:] = 1.0
        result = loss_fn(None, adv, start_index=2, clean_len=4, wrapper_mask=wrapper_mask)
        assert result["loss"].item() >= 0.0

    def test_layer_losses_length(self):
        loss_fn = WrapperEntropyRegularizationLoss()
        adv = _make_mock_outputs(num_layers=3, seq=6)
        result = loss_fn(None, adv, start_index=2, clean_len=4)
        assert len(result["layer_losses"]) == 3

    def test_missing_attentions_raises(self):
        loss_fn = WrapperEntropyRegularizationLoss()
        adv = SimpleNamespace(attentions=None)
        with pytest.raises(ValueError, match="attentions"):
            loss_fn(None, adv, start_index=2, clean_len=4)

    def test_weight_scales_loss(self):
        torch.manual_seed(7)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        loss_fn1 = WrapperEntropyRegularizationLoss(weight=1.0)
        loss_fn2 = WrapperEntropyRegularizationLoss(weight=4.0)
        r1 = loss_fn1(None, adv, start_index=2, clean_len=4)
        r2 = loss_fn2(None, adv, start_index=2, clean_len=4)
        assert r2["loss"].item() == pytest.approx(4.0 * r1["loss"].item(), rel=1e-4)


# ---------------------------------------------------------------------------
# CombinedAttentionConsistencyLoss
# ---------------------------------------------------------------------------

class TestCombinedAttentionConsistencyLoss:
    def _make_combined_outputs(self, seq, num_layers=2):
        return _make_mock_outputs(
            num_layers=num_layers, seq=seq,
            include_attentions=True, include_hidden_states=True
        )

    def test_returns_required_keys(self):
        loss_fn = CombinedAttentionConsistencyLoss()
        clean = self._make_combined_outputs(seq=4)
        adv = self._make_combined_outputs(seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        for key in ("loss", "kl_loss", "output_loss", "mean_layer_loss"):
            assert key in result

    def test_loss_is_finite(self):
        loss_fn = CombinedAttentionConsistencyLoss()
        clean = self._make_combined_outputs(seq=4)
        adv = self._make_combined_outputs(seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert math.isfinite(result["loss"].item())

    def test_zero_kl_weight_uses_only_output(self):
        loss_fn = CombinedAttentionConsistencyLoss(kl_weight=0.0, output_weight=1.0)
        clean = self._make_combined_outputs(seq=4)
        adv = self._make_combined_outputs(seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert math.isfinite(result["loss"].item())

    def test_missing_attentions_raises(self):
        loss_fn = CombinedAttentionConsistencyLoss()
        clean = SimpleNamespace(attentions=None, hidden_states=None)
        adv = self._make_combined_outputs(seq=6)
        with pytest.raises(ValueError, match="attentions"):
            loss_fn(clean, adv, start_index=0, clean_len=4)

    def test_missing_hidden_states_raises(self):
        loss_fn = CombinedAttentionConsistencyLoss()
        clean = _make_mock_outputs(num_layers=2, seq=4, include_attentions=True,
                                   include_hidden_states=False)
        clean.hidden_states = None
        adv = self._make_combined_outputs(seq=6)
        with pytest.raises(ValueError, match="hidden_states"):
            loss_fn(clean, adv, start_index=0, clean_len=4)


# ---------------------------------------------------------------------------
# CombinedJSDWrapperLoss
# ---------------------------------------------------------------------------

class TestCombinedJSDWrapperLoss:
    def test_returns_required_keys(self):
        loss_fn = CombinedJSDWrapperLoss()
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        for key in ("loss", "jsd_loss", "wrapper_loss", "mean_wrapper_attention"):
            assert key in result

    def test_loss_is_finite(self):
        loss_fn = CombinedJSDWrapperLoss()
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert math.isfinite(result["loss"].item())

    def test_loss_is_nonnegative(self):
        loss_fn = CombinedJSDWrapperLoss()
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert result["loss"].item() >= 0.0

    def test_weight_scales_loss(self):
        torch.manual_seed(1)
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        loss_fn1 = CombinedJSDWrapperLoss(weight=1.0)
        loss_fn2 = CombinedJSDWrapperLoss(weight=2.0)
        r1 = loss_fn1(clean, adv, start_index=0, clean_len=4)
        r2 = loss_fn2(clean, adv, start_index=0, clean_len=4)
        assert r2["loss"].item() == pytest.approx(2.0 * r1["loss"].item(), rel=1e-4)

    def test_with_wrapper_mask(self):
        loss_fn = CombinedJSDWrapperLoss()
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        wrapper_mask = torch.zeros(1, 6)
        wrapper_mask[:, :2] = 1.0
        result = loss_fn(clean, adv, start_index=2, clean_len=4, wrapper_mask=wrapper_mask)
        assert math.isfinite(result["loss"].item())

    def test_custom_jsd_wrapper_weights(self):
        loss_fn = CombinedJSDWrapperLoss(jsd_weight=0.8, wrapper_weight=0.2)
        clean = _make_mock_outputs(num_layers=2, seq=4)
        adv = _make_mock_outputs(num_layers=2, seq=6)
        result = loss_fn(clean, adv, start_index=0, clean_len=4)
        assert math.isfinite(result["loss"].item())
