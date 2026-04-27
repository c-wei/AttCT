"""Unit tests for consistency loss functions, focused on ActivationConsistencyLoss.

Run with:
    uv run python -m pytest losses/test_losses.py -v
"""

from types import SimpleNamespace
from typing import Tuple

import pytest
import torch

from losses.losses import ActivationConsistencyLoss


def _make_outputs(hidden_states: Tuple[torch.Tensor, ...]) -> SimpleNamespace:
    """Build a fake HF-style ModelOutput holding `hidden_states`."""
    return SimpleNamespace(hidden_states=hidden_states, attentions=None)


def _random_hs(num_layers: int, batch: int, seq_len: int, hidden_dim: int) -> Tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    return tuple(torch.randn(batch, seq_len, hidden_dim) for _ in range(num_layers + 1))


# ─── Identical activations → zero loss ────────────────────────────────────────

class TestZeroLoss:
    def test_identical_hidden_states_give_zero_loss(self):
        hs = _random_hs(num_layers=4, batch=1, seq_len=8, hidden_dim=16)
        out = _make_outputs(hs)

        loss_fn = ActivationConsistencyLoss(weight=1.0)
        result = loss_fn(out, out, start_index=0, clean_len=8, match_len=8)

        assert result["loss"].item() == 0.0
        assert all(l == 0.0 for l in result["layer_losses"])

    def test_identical_hidden_states_with_mse_formulation(self):
        hs = _random_hs(num_layers=4, batch=1, seq_len=8, hidden_dim=16)
        out = _make_outputs(hs)

        loss_fn = ActivationConsistencyLoss(weight=1.0, loss_formulation="mse")
        result = loss_fn(out, out, start_index=0, clean_len=8, match_len=8)

        assert result["loss"].item() == 0.0


# ─── Embedding layer is excluded by default ───────────────────────────────────

class TestEmbeddingLayerExclusion:
    def test_default_skips_layer_zero(self):
        hs = _random_hs(num_layers=3, batch=1, seq_len=4, hidden_dim=8)
        out = _make_outputs(hs)

        loss_fn = ActivationConsistencyLoss(weight=1.0)
        result = loss_fn(out, out, start_index=0, clean_len=4, match_len=4)

        # 3 transformer layers → 3 entries in layer_losses (not 4)
        assert result["num_layers_used"] == 3
        assert len(result["layer_losses"]) == 3

    def test_all_with_embedding_includes_layer_zero(self):
        hs = _random_hs(num_layers=3, batch=1, seq_len=4, hidden_dim=8)
        out = _make_outputs(hs)

        loss_fn = ActivationConsistencyLoss(weight=1.0, layer_selection="all_with_embedding")
        result = loss_fn(out, out, start_index=0, clean_len=4, match_len=4)

        assert result["num_layers_used"] == 4   # embedding + 3 layers


# ─── Paper formulation matches the formula by hand ────────────────────────────

class TestPaperFormulation:
    def test_paper_loss_equals_l2_squared_per_token(self):
        # 1 layer (so num_layers=1, hidden_states has 2 entries: emb + 1 layer).
        # Pick a deterministic 1×T×D shift between clean and adv to verify the
        # exact paper formula: sum over D of (h_w - h_c)², averaged over T.
        torch.manual_seed(42)
        T, D = 5, 3
        clean_layer  = torch.randn(1, T, D)
        delta        = torch.randn(1, T, D)
        adv_layer    = clean_layer + delta

        # Fill embedding with anything — it's skipped by default.
        emb = torch.zeros(1, T, D)

        clean_out = _make_outputs((emb, clean_layer))
        adv_out   = _make_outputs((emb, adv_layer))

        loss_fn = ActivationConsistencyLoss(weight=1.0, loss_formulation="paper")
        result = loss_fn(clean_out, adv_out, start_index=0, clean_len=T, match_len=T)

        # Expected: mean_t [ sum_d (delta_t,d)² ]
        expected = (delta ** 2).sum(dim=-1).mean().item()
        assert result["loss"].item() == pytest.approx(expected, rel=1e-6)

    def test_paper_loss_is_d_times_mse_loss(self):
        # For the same inputs, the paper formulation should equal D * mse formulation.
        torch.manual_seed(7)
        T, D = 4, 6
        clean_layer = torch.randn(1, T, D)
        adv_layer   = torch.randn(1, T, D)
        emb = torch.zeros(1, T, D)

        clean_out = _make_outputs((emb, clean_layer))
        adv_out   = _make_outputs((emb, adv_layer))

        paper = ActivationConsistencyLoss(weight=1.0, loss_formulation="paper")
        mse   = ActivationConsistencyLoss(weight=1.0, loss_formulation="mse")

        p = paper(clean_out, adv_out, start_index=0, clean_len=T, match_len=T)["loss"].item()
        m = mse(clean_out, adv_out,   start_index=0, clean_len=T, match_len=T)["loss"].item()

        assert p == pytest.approx(D * m, rel=1e-5)


# ─── Matching window: match_len takes precedence over clean_len ───────────────

class TestMatchLenSlicing:
    def test_match_len_slices_from_end_of_each_sequence(self):
        # Clean has length 5, wrapped has length 10. match_len=3 → compare the
        # last 3 positions of each.
        torch.manual_seed(0)
        D = 4
        clean_emb = torch.zeros(1, 5, D)
        adv_emb   = torch.zeros(1, 10, D)

        # Layer 1: arrange so that the LAST 3 tokens of clean equal the LAST 3
        # tokens of adv but the rest differ. Then match_len=3 → loss should be 0.
        clean_layer = torch.randn(1, 5, D)
        adv_layer   = torch.randn(1, 10, D)
        adv_layer[:, -3:, :] = clean_layer[:, -3:, :]

        clean_out = _make_outputs((clean_emb, clean_layer))
        adv_out   = _make_outputs((adv_emb, adv_layer))

        loss_fn = ActivationConsistencyLoss(weight=1.0)
        result = loss_fn(clean_out, adv_out, match_len=3)

        assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)
        assert result["match_len"] == 3

    def test_legacy_mode_uses_clean_len_when_no_match_len(self):
        # Without match_len, fall back to start_index/clean_start_index/clean_len.
        torch.manual_seed(0)
        D = 4
        # Both prompts length 6, content at different absolute positions.
        clean_layer = torch.randn(1, 6, D)
        adv_layer   = torch.randn(1, 6, D)
        # Make positions [0:3] match in clean and [2:5] in wrapped.
        adv_layer[:, 2:5, :] = clean_layer[:, 0:3, :]

        emb = torch.zeros(1, 6, D)
        clean_out = _make_outputs((emb, clean_layer))
        adv_out   = _make_outputs((emb, adv_layer))

        loss_fn = ActivationConsistencyLoss(weight=1.0)
        result = loss_fn(
            clean_out, adv_out,
            start_index=2, clean_start_index=0, clean_len=3,
        )

        assert result["loss"].item() == pytest.approx(0.0, abs=1e-6)


# ─── Stop-gradient on clean side ──────────────────────────────────────────────

class TestStopGradient:
    def test_clean_side_does_not_receive_gradient(self):
        # Clean tensor requires_grad; loss.backward() should leave its .grad as None.
        D, T = 5, 3
        clean_layer = torch.randn(1, T, D, requires_grad=True)
        adv_layer   = torch.randn(1, T, D, requires_grad=True)
        emb = torch.zeros(1, T, D)

        clean_out = _make_outputs((emb, clean_layer))
        adv_out   = _make_outputs((emb, adv_layer))

        loss_fn = ActivationConsistencyLoss(weight=1.0)
        loss = loss_fn(clean_out, adv_out, match_len=T)["loss"]
        loss.backward()

        assert clean_layer.grad is None
        assert adv_layer.grad is not None
        assert adv_layer.grad.abs().sum().item() > 0


# ─── Weight scaling ───────────────────────────────────────────────────────────

class TestWeightScaling:
    def test_weight_scales_loss_linearly(self):
        torch.manual_seed(1)
        D, T = 4, 3
        clean_layer = torch.randn(1, T, D)
        adv_layer   = torch.randn(1, T, D)
        emb = torch.zeros(1, T, D)

        clean_out = _make_outputs((emb, clean_layer))
        adv_out   = _make_outputs((emb, adv_layer))

        loss_w1   = ActivationConsistencyLoss(weight=1.0)(
            clean_out, adv_out, match_len=T)["loss"].item()
        loss_w7   = ActivationConsistencyLoss(weight=7.0)(
            clean_out, adv_out, match_len=T)["loss"].item()

        assert loss_w7 == pytest.approx(7 * loss_w1, rel=1e-5)


# ─── Edge cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_zero_match_len_returns_zero_loss(self):
        torch.manual_seed(0)
        D, T = 4, 3
        clean_layer = torch.randn(1, T, D)
        adv_layer   = torch.randn(1, T, D)
        emb = torch.zeros(1, T, D)

        clean_out = _make_outputs((emb, clean_layer))
        adv_out   = _make_outputs((emb, adv_layer))

        loss_fn = ActivationConsistencyLoss(weight=1.0)
        result = loss_fn(clean_out, adv_out, match_len=0, clean_len=0)

        assert result["loss"].item() == 0.0

    def test_missing_hidden_states_raises(self):
        out = SimpleNamespace(hidden_states=None, attentions=None)
        loss_fn = ActivationConsistencyLoss(weight=1.0)

        with pytest.raises(ValueError, match="hidden_states"):
            loss_fn(out, out, match_len=4)

    def test_invalid_loss_formulation_raises(self):
        with pytest.raises(ValueError, match="loss_formulation"):
            ActivationConsistencyLoss(loss_formulation="bogus")
