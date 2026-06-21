"""Turnkey multi-objective reranker built on the RLProp core.

This is the cleaned-up successor of the original ``morsify`` driver: give it a
candidate pool, a few objective value functions, and target ratios, and it builds
the marginal gains for you (re-computing them at every step so non-additive
objectives like diversity work correctly), optionally normalizes them, and runs
the RLProp allocation.

It is convenient but, by definition, more expensive than :meth:`rlprop.RLProp.select`
(it probes every candidate for every objective at every step). For additive
objectives, precompute marginal gains once and use ``RLProp.select`` instead.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .core import RLProp, StepFn
from .normalize import normalize_mgains

ObjectiveFn = Callable[[Sequence[int]], float]


def rerank(
    candidates,
    objectives: Sequence[ObjectiveFn],
    weights: Sequence[float] | np.ndarray,
    k: int,
    *,
    normalize: str | None = None,
    allocator: str | StepFn = "rlprop",
    tie_breaking: float = 0.0,
    exclude=None,
) -> np.ndarray:
    """Rerank ``candidates`` into a ``k``-item list proportional to ``weights``.

    Parameters
    ----------
    candidates : 1-D array-like of candidate item ids (the pool to rerank). Build
        this however you like — e.g. the top-N by relevance from your recommender.
    objectives : objective value functions ``f(selected_item_ids) -> float``. See
        :mod:`rlprop.objectives` for ready-made relevance / diversity / novelty.
    weights : target ratios over the objectives, e.g. ``[0.6, 0.3, 0.1]``.
    k : length of the returned list.
    normalize : per-objective marginal-gain normalization applied at each step;
        one of ``None``/``"none"``, ``"minmax"``, ``"standard"``, ``"robust"``,
        ``"quantile"``.
    allocator : allocation rule, see :class:`rlprop.RLProp`.
    tie_breaking : optional fairness nudge on near-ties.
    exclude : item ids to never select.

    Returns
    -------
    np.ndarray of selected item ids, in selection order.
    """
    candidates = np.asarray(candidates)
    if candidates.ndim != 1:
        raise ValueError("candidates must be 1-D.")
    n_obj = len(objectives)
    if n_obj == 0:
        raise ValueError("provide at least one objective.")

    engine = RLProp(weights, allocator=allocator, tie_breaking=tie_breaking)
    if engine.weights.size != n_obj:
        raise ValueError(f"{n_obj} objectives but {engine.weights.size} weights.")

    n = candidates.size
    available = np.ones(n, dtype=bool)
    if exclude is not None:
        excluded = set(int(e) for e in exclude)
        for pos, item in enumerate(candidates):
            if int(item) in excluded:
                available[pos] = False

    selected_items: list[int] = []
    budget = min(int(k), int(available.sum()))

    for _ in range(budget):
        mgains = np.zeros((n_obj, n), dtype=float)
        for o, f in enumerate(objectives):
            base = f(selected_items)
            for pos in range(n):
                if available[pos]:
                    mgains[o, pos] = f(selected_items + [int(candidates[pos])]) - base
        mgains = normalize_mgains(mgains, normalize)
        pos = engine.step(mgains, available)
        available[pos] = False
        selected_items.append(int(candidates[pos]))

    return np.asarray(selected_items)
