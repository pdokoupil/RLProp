"""Lock RLProp's behavior to the original (corrected) implementation.

The reference here is the fully-vectorized single-session step used in the LTP
code base (``fast_impl_v3_without_history``), which is the correct form of the
original ``greedy_max`` allocator (the version in
``exactly_proportional_fuzzy_dhondt_2.py`` carried a refactor bug in its per-item
loop). If this test ever breaks, the core math has drifted.
"""

import numpy as np

from rlprop import RLProp

NEG_INF = int(-10e6)


def _reference_rlprop(mgains, weights, k):
    """Direct re-implementation of the legacy vectorized RLProp loop."""
    group_size, n_items = mgains.shape
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()

    rating_matrix_scaled = mgains.astype(np.float64)
    TOT = 0.0
    gm = np.zeros(group_size, dtype=np.float64)
    seen_items_mask = np.ones(n_items, dtype=np.float64)

    positive_gain_mask = rating_matrix_scaled >= 0.0
    negative_gain_mask = rating_matrix_scaled < 0.0

    top_k = []
    for _ in range(k):
        gain_items = np.zeros_like(rating_matrix_scaled)
        tots = np.maximum(TOT, TOT + rating_matrix_scaled.sum(axis=0))
        remainder = tots * weights[:, np.newaxis] - gm[:, np.newaxis]
        gain_items[positive_gain_mask] = np.maximum(
            0, np.minimum(rating_matrix_scaled, remainder)[positive_gain_mask]
        )
        gain_items[negative_gain_mask] = np.minimum(
            0, (rating_matrix_scaled - remainder)[negative_gain_mask]
        )
        scores = gain_items.sum(axis=0)
        scores = scores * seen_items_mask + NEG_INF * (1 - seen_items_mask)
        i_best = int(scores.argmax())
        seen_items_mask[i_best] = 0
        gm = gm + rating_matrix_scaled[:, i_best]
        TOT = np.clip(gm, 0.0, None).sum()
        top_k.append(i_best)
    return top_k


def test_matches_reference_positive_gains():
    rng = np.random.default_rng(7)
    mgains = rng.random((3, 80))
    w = [0.6, 0.3, 0.1]
    assert RLProp(w).select(mgains, k=15).tolist() == _reference_rlprop(mgains, w, 15)


def test_matches_reference_with_negative_gains():
    rng = np.random.default_rng(8)
    mgains = rng.normal(size=(4, 60))   # includes negative marginal gains
    w = [0.4, 0.3, 0.2, 0.1]
    assert RLProp(w).select(mgains, k=12).tolist() == _reference_rlprop(mgains, w, 12)
