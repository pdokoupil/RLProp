"""Core RLProp allocation: pick k items proportionally across objectives.

The algorithm is the mandate-allocation / fuzzy-D'Hondt step from
Peska & Dokoupil, *Towards Results-level Proportionality for Multi-objective
Recommender Systems* (SIGIR '22). Given per-item marginal gains for each
objective and target weights over objectives, it greedily selects the item that
best restores proportionality between accumulated gains and the target weights.

Everything here is single-list and pure NumPy.
"""

from __future__ import annotations

from typing import Callable, Iterable, Sequence

import numpy as np

# Sentinel score for masked-out (unavailable / already-selected) items.
_NEG_INF = -1e12

# A single allocation step: given current state, return the index of the next item.
StepFn = Callable[[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray], int]


def _mask_scores(scores: np.ndarray, available: np.ndarray) -> np.ndarray:
    """Push unavailable items to the bottom (cf. the original ``mask_scores``)."""
    return np.where(available, scores, _NEG_INF)


def rlprop_step(
    mgains: np.ndarray,
    gm: np.ndarray,
    tot: float,
    weights: np.ndarray,
    available: np.ndarray,
    tie_breaking: float = 0.0,
) -> int:
    """One RLProp / fuzzy-D'Hondt step (the corrected, fully-vectorized version).

    Parameters
    ----------
    mgains : (n_objectives, n_items) array of per-item marginal gains.
    gm : (n_objectives,) accumulated gain per objective so far.
    tot : running total of (clipped) accumulated gains.
    weights : (n_objectives,) target weights, assumed already normalized to sum 1.
    available : (n_items,) bool, True where an item may still be picked.
    tie_breaking : optional LTP-style nudge toward fairness on near-ties.
    """
    # Prospective total if this item were added (per item), never below current tot.
    tots = np.maximum(tot, tot + mgains.sum(axis=0))                 # (n_items,)
    # Per-objective "unused mandate" the item could fill.
    remainder = tots[None, :] * weights[:, None] - gm[:, None]       # (n_obj, n_items)

    gain = np.empty_like(mgains)
    pos = mgains >= 0.0
    gain[pos] = np.maximum(0.0, np.minimum(mgains, remainder)[pos])
    gain[~pos] = np.minimum(0.0, (mgains - remainder)[~pos])
    scores = gain.sum(axis=0)                                        # (n_items,)

    if tie_breaking:
        tie = remainder <= mgains
        bonus = np.zeros_like(mgains)
        bonus[tie] = tie_breaking * (mgains[tie] - remainder[tie])
        scores = scores + bonus.sum(axis=0)

    return int(_mask_scores(scores, available).argmax())


def fuzzy_dhondt_step(
    mgains: np.ndarray,
    gm: np.ndarray,
    tot: float,
    weights: np.ndarray,
    available: np.ndarray,
    tie_breaking: float = 0.0,
) -> int:
    """RLProp step that breaks ties toward items with the larger prospective total.

    Mirrors ``exactly_proportional_fuzzy_dhondt_2`` from the original code base.
    """
    tots = np.maximum(tot, tot + mgains.sum(axis=0))
    remainder = tots[None, :] * weights[:, None] - gm[:, None]
    gain = np.empty_like(mgains)
    pos = mgains >= 0.0
    gain[pos] = np.maximum(0.0, np.minimum(mgains, remainder)[pos])
    gain[~pos] = np.minimum(0.0, (mgains - remainder)[~pos])
    scores = _mask_scores(gain.sum(axis=0), available)

    best = scores.max()
    # Among max-gain items, prefer the one with the largest prospective total.
    candidates = np.where(scores >= best, tots, _NEG_INF)
    return int(candidates.argmax())


def weighted_average_step(
    mgains: np.ndarray,
    gm: np.ndarray,
    tot: float,
    weights: np.ndarray,
    available: np.ndarray,
    tie_breaking: float = 0.0,
) -> int:
    """Baseline: pick the item maximizing the weighted sum of marginal gains.

    This is *not* proportional — it's the standard weighted-scalarization baseline,
    included so users can compare against RLProp in one line.
    """
    scores = (mgains * weights[:, None]).sum(axis=0)
    return int(_mask_scores(scores, available).argmax())


ALLOCATORS: dict[str, StepFn] = {
    "rlprop": rlprop_step,
    "fuzzy_dhondt": fuzzy_dhondt_step,
    "weighted_average": weighted_average_step,
}


def _resolve_allocator(allocator: str | StepFn) -> StepFn:
    if callable(allocator):
        return allocator
    try:
        return ALLOCATORS[allocator]
    except KeyError:
        raise ValueError(
            f"Unknown allocator {allocator!r}. "
            f"Choose one of {sorted(ALLOCATORS)} or pass a callable."
        ) from None


def _as_weights(weights: Sequence[float] | np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=float).ravel()
    if w.ndim != 1 or w.size == 0:
        raise ValueError("weights must be a non-empty 1-D sequence.")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative.")
    s = w.sum()
    if s <= 0:
        raise ValueError("weights must sum to a positive value.")
    return w / s


class RLProp:
    """Select ``k`` items proportionally across objectives from marginal gains.

    Use this when you already have a marginal-gain matrix. If your objectives are
    additive (e.g. relevance, popularity-based novelty), the marginal gains are
    static and one ``select`` call does the whole job. For objectives whose
    marginal gain depends on what's already chosen (e.g. intra-list diversity),
    use :func:`rlprop.rerank` instead, or drive :meth:`step` yourself.

    Parameters
    ----------
    weights : target ratios over objectives, e.g. ``[0.6, 0.3, 0.1]``. Normalized
        internally, so ``[6, 3, 1]`` is equivalent.
    allocator : ``"rlprop"`` (default), ``"fuzzy_dhondt"``, ``"weighted_average"``,
        or any callable with the :data:`StepFn` signature.
    tie_breaking : optional fairness nudge passed to the step function.
    """

    def __init__(
        self,
        weights: Sequence[float] | np.ndarray,
        allocator: str | StepFn = "rlprop",
        tie_breaking: float = 0.0,
    ) -> None:
        self.weights = _as_weights(weights)
        self.allocator_name = allocator if isinstance(allocator, str) else getattr(allocator, "__name__", "custom")
        self._step = _resolve_allocator(allocator)
        self.tie_breaking = float(tie_breaking)
        self.reset()

    def reset(self) -> "RLProp":
        """Clear accumulated state so the next selection starts fresh."""
        self.gm = np.zeros_like(self.weights)
        self.tot = 0.0
        return self

    def step(self, mgains: np.ndarray, available: np.ndarray) -> int:
        """Run one allocation step and update internal state with the winner."""
        mgains = np.asarray(mgains, dtype=float)
        i = self._step(mgains, self.gm, self.tot, self.weights, available, self.tie_breaking)
        self.gm = self.gm + mgains[:, i]
        self.tot = float(np.clip(self.gm, 0.0, None).sum())
        return i

    def select(
        self,
        mgains: np.ndarray,
        k: int,
        exclude: Iterable[int] | None = None,
    ) -> np.ndarray:
        """Select ``k`` item indices from a static marginal-gain matrix.

        Parameters
        ----------
        mgains : (n_objectives, n_items) marginal gains. Accepts anything
            array-like (NumPy, pandas, CPU torch) via ``np.asarray``.
        k : number of items to return.
        exclude : item indices to never select (e.g. already-seen items).

        Returns
        -------
        np.ndarray of selected item indices, in selection order.
        """
        mgains = np.asarray(mgains, dtype=float)
        if mgains.ndim != 2:
            raise ValueError(f"mgains must be 2-D (n_objectives, n_items); got shape {mgains.shape}.")
        n_obj, n_items = mgains.shape
        if n_obj != self.weights.size:
            raise ValueError(
                f"mgains has {n_obj} objectives but {self.weights.size} weights were given."
            )

        available = np.ones(n_items, dtype=bool)
        if exclude is not None:
            available[np.asarray(list(exclude), dtype=int)] = False

        self.reset()
        budget = min(int(k), int(available.sum()))
        chosen = np.empty(budget, dtype=np.int64)
        for pos in range(budget):
            i = self.step(mgains, available)
            chosen[pos] = i
            available[i] = False
        return chosen
