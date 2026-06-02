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
    neutral_acts: np.ndarray,                  # (n_neutral, hidden_dim) neutral dialogues
) -> dict:
    """Run the four-check gauntlet. All means are role-level means (one per role
    role-mean over its question rollouts)."""
    anchor_proj = _project(assistant_anchor_acts, axis)
    other_proj  = _project(other_acts, axis)
    pos_proj    = _project(sanity_pos_acts, axis)
    neg_proj    = _project(sanity_neg_acts, axis)
    neut_proj   = _project(neutral_acts, axis)

    anchor_mean = float(anchor_proj.mean())
    other_mean  = float(other_proj.mean())
    other_std   = float(other_proj.std(ddof=1))
    median_role_proj = float(np.median(np.concatenate([anchor_proj, other_proj])))

    # 1. Self-consistency
    g1 = anchor_mean >= abs(median_role_proj)

    # 2. Held-out negatives: each should be bottom-3 across all role projections
    all_roles_with_neg = np.concatenate([anchor_proj, other_proj, neg_proj])
    sorted_idx = np.argsort(all_roles_with_neg)  # ascending
    n_total = all_roles_with_neg.size
    n_neg = neg_proj.size
    # neg roles occupy indices [n_total - n_neg ... n_total - 1] of the concatenated array
    neg_start = anchor_proj.size + other_proj.size
    neg_indices_in_concat = set(range(neg_start, neg_start + n_neg))
    bottom_3 = set(sorted_idx[:3].tolist())
    # Pass if at least 2 of 3 held-out negatives land in the bottom-3
    n_neg_in_bottom3 = len(neg_indices_in_concat & bottom_3)
    g2 = n_neg_in_bottom3 >= 2
    # Strength check: each neg should be >=1 SD below `other` mean
    neg_below_1sd = all((p < other_mean - other_std) for p in neg_proj)
    g2 = g2 and neg_below_1sd

    # 3. Held-out positives: mildly positive, below anchor mean
    pos_below_anchor = all(float(p) < anchor_mean for p in pos_proj)
    pos_above_other  = float(pos_proj.mean()) > other_mean
    g3 = pos_below_anchor and pos_above_other

    # 4. Topic-confound: neutral spread should be small vs role spread
    role_spread = float(np.concatenate([anchor_proj, other_proj]).std(ddof=1))
    neutral_spread = float(neut_proj.std(ddof=1)) if neut_proj.size > 1 else 0.0
    signal_to_noise = role_spread / max(neutral_spread, 1e-9)
    g4 = signal_to_noise > 2.0

    return {
        "anchor_mean":      anchor_mean,
        "other_mean":       other_mean,
        "other_std":        other_std,
        "median_role_proj": median_role_proj,
        "pos_projections":  pos_proj.tolist(),
        "neg_projections":  neg_proj.tolist(),
        "neutral_spread":   neutral_spread,
        "role_spread":      role_spread,
        "signal_to_noise":  signal_to_noise,
        "g1_self_consistency":    bool(g1),
        "g2_negatives_bottom3":   bool(g2),
        "g3_positives_below_anchor": bool(g3),
        "g4_neutral_spread_low":  bool(g4),
        "all_pass":               bool(g1 and g2 and g3 and g4),
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
