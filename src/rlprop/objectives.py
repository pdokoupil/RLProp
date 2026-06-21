"""Ready-made objective value functions for :func:`rlprop.rerank`.

An *objective value function* takes the list of currently-selected item ids and
returns a single scalar — the value of that (partial) list for the objective.
``rerank`` turns these into marginal gains by probing ``f(L + [i]) - f(L)``.

These three cover the relevance / diversity / novelty setup from the paper. They
are conveniences: writing your own is just a one-line function. For speed on
large candidate pools, prefer additive objectives (relevance, novelty) and
:meth:`rlprop.RLProp.select`, since their marginal gains are static.

In all of them, item ids are assumed to index into the provided arrays/matrix.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

ObjectiveFn = Callable[[Sequence[int]], float]


def relevance(scores) -> ObjectiveFn:
    """Sum of per-item relevance scores. Additive (static marginal gains)."""
    scores = np.asarray(scores, dtype=float)

    def f(items: Sequence[int]) -> float:
        items = list(items)
        return float(scores[items].sum()) if items else 0.0

    return f


def novelty(popularity) -> ObjectiveFn:
    """Mean popularity-complement (1 - normalized popularity). Additive."""
    pop = np.asarray(popularity, dtype=float)
    pmax = pop.max()
    norm = pop / pmax if pmax > 0 else pop

    def f(items: Sequence[int]) -> float:
        items = list(items)
        return float(np.mean(1.0 - norm[items])) if items else 0.0

    return f


def diversity(distance_matrix) -> ObjectiveFn:
    """Intra-list diversity: mean pairwise distance among selected items.

    Non-additive — the marginal gain depends on what is already selected — so this
    one genuinely needs :func:`rlprop.rerank` rather than ``RLProp.select``.
    """
    dist = np.asarray(distance_matrix, dtype=float)

    def f(items: Sequence[int]) -> float:
        items = list(items)
        if len(items) < 2:
            return 0.0
        idx = np.asarray(items)
        sub = dist[np.ix_(idx, idx)]
        iu = np.triu_indices(len(items), k=1)
        return float(sub[iu].mean())

    return f
