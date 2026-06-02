"""Sanity gauntlet for a candidate Assistant Axis at a given layer.

A layer "graduates" if all four checks pass:

  1. Self-consistency: assistant-anchor mean projects strongly positive.
  2. Held-out negative roles (corpse / eldritch / revenant) rank in the
     bottom-3 of all roles + held-outs (≥1 SD below extraction-set mean).
  3. Held-out positive roles (tutor / instructor) project mildly positive,
     below the extraction-set assistant-anchor mean.
  4. Topic-confound check: neutral-dialogue projection spread is small
     vs role-projection spread (signal-to-noise > 2).

Returns a single JSON-serialisable dict per layer with pass/fail flags +
the underlying numbers for inspection.
"""
from __future__ import annotations

import numpy as np


def _project(activations: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """activations: (N, hidden). axis: (hidden,). Returns (N,)."""
    return activations @ axis


def evaluate_layer(
    *,
    axis: np.ndarray,                          # (hidden_dim,) normalised contrast vector
    assistant_anchor_acts: np.ndarray,         # (n_anchor, hidden_dim) means per anchor role
    other_acts: np.ndarray,                    # (n_other, hidden_dim) means per other role
    sanity_pos_acts: np.ndarray,               # (n_pos_held, hidden_dim) held-out positive
    sanity_neg_acts: np.ndarray,               # (n_neg_held, hidden_dim) held-out negative
    neutral_acts: np.ndarray = None,           # (unused; kept for backward-compat)
) -> dict:
    """Run the four-check gauntlet. All means are role-level means (one per role).

    Revised checks (post-v1):
      g1 (self-consistency):    anchor_mean is above the role median (relative).
      g2 (negatives at bottom): held-out negatives land in the bottom-3 of all
                                roles+negs AND each is >=1 std below the other_mean.
      g3 (positives at top):    held-out positives project above the other_mean
                                (they don't have to be below anchor — being MORE
                                assistant-like than in-set anchors is good news).
      g4 (effect size):         (anchor_mean - other_mean) / other_std > 1.0
                                — replaces the prior neutral-spread check, which
                                was measuring format mismatch (neutral_dialogues
                                aren't chat-formatted) not axis quality.
    """
    anchor_proj = _project(assistant_anchor_acts, axis)
    other_proj  = _project(other_acts, axis)
    pos_proj    = _project(sanity_pos_acts, axis)
    neg_proj    = _project(sanity_neg_acts, axis)

    anchor_mean = float(anchor_proj.mean())
    other_mean  = float(other_proj.mean())
    other_std   = float(other_proj.std(ddof=1))
    median_role_proj = float(np.median(np.concatenate([anchor_proj, other_proj])))

    # g1: relative — anchor above role median
    g1 = anchor_mean > median_role_proj

    # g2: at least 2/3 held-out negatives in bottom-3 of all (roles + negs)
    #     AND each held-out neg is >=1 std below the other_mean
    all_roles_with_neg = np.concatenate([anchor_proj, other_proj, neg_proj])
    sorted_idx = np.argsort(all_roles_with_neg)  # ascending
    neg_start = anchor_proj.size + other_proj.size
    neg_indices_in_concat = set(range(neg_start, neg_start + neg_proj.size))
    bottom_3 = set(sorted_idx[:3].tolist())
    n_neg_in_bottom3 = len(neg_indices_in_concat & bottom_3)
    g2_count = n_neg_in_bottom3 >= 2
    g2_strength = all((p < other_mean - other_std) for p in neg_proj)
    g2 = g2_count and g2_strength

    # g3: held-out positives project above the other_mean (Assistant-aligned)
    g3 = float(pos_proj.mean()) > other_mean

    # g4: effect size — anchor mean above other mean by >=1 std
    effect_size = (anchor_mean - other_mean) / max(other_std, 1e-9)
    g4 = effect_size > 1.0

    return {
        "anchor_mean":      anchor_mean,
        "other_mean":       other_mean,
        "other_std":        other_std,
        "median_role_proj": median_role_proj,
        "pos_projections":  pos_proj.tolist(),
        "neg_projections":  neg_proj.tolist(),
        "effect_size":      effect_size,
        "g1_anchor_above_median":  bool(g1),
        "g2_negatives_bottom3":    bool(g2),
        "g3_positives_above_other": bool(g3),
        "g4_effect_size_over_1":   bool(g4),
        "all_pass":                bool(g1 and g2 and g3 and g4),
    }


def compute_contrast_axis(
    assistant_anchor_acts: np.ndarray,   # (n_anchor, hidden_dim)
    other_acts: np.ndarray,              # (n_other, hidden_dim)
) -> np.ndarray:
    """axis = mean(assistant_anchor) - mean(other), normalised."""
    a = assistant_anchor_acts.mean(axis=0)
    b = other_acts.mean(axis=0)
    v = a - b
    return v / (np.linalg.norm(v) + 1e-9)
