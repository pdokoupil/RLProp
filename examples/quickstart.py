"""RLProp quickstart — runs in well under a second, pure NumPy, no data download.

Shows both ways in:
  1. RLProp.select  -- you already have marginal gains
  2. rerank         -- you have objective functions (relevance + diversity)
"""

import numpy as np

from rlprop import RLProp, rerank
from rlprop import objectives as obj


def proportions(selected, mgains):
    """How much of each objective's total gain the selected list captured."""
    got = mgains[:, selected].sum(axis=1)
    return got / got.sum()


def main() -> None:
    rng = np.random.default_rng(0)

    # --- 1) Primitive: bring your own marginal gains -------------------------
    # Three objectives over 60 items, where each item mostly serves ONE objective
    # (20 items per specialty). With this structure you can clearly see the
    # selected list's composition track the requested ratios.
    n_per, n_obj = 20, 3
    mgains = 0.05 * rng.random((n_obj, n_per * n_obj))
    for c in range(n_obj):
        mgains[c, c * n_per:(c + 1) * n_per] += 1.0

    print("target weights  ->  achieved proportions over the selected top-10")
    for w in ([1, 0, 0], [0.6, 0.3, 0.1], [0.34, 0.33, 0.33]):
        top_k = RLProp(w).select(mgains, k=10)
        print(f"  {w}  ->  {np.round(proportions(top_k, mgains), 2)}")

    # --- 2) Turnkey reranker: objective functions ---------------------------
    n_items = 60
    rel = rng.random(n_items)                       # base relevance from your model
    dist = rng.random((n_items, n_items))           # item-item distance matrix
    dist = (dist + dist.T) / 2                       # make it symmetric
    np.fill_diagonal(dist, 0.0)

    candidates = np.argsort(-rel)[:30]              # rerank the top-30 by relevance

    recs = rerank(
        candidates=candidates,
        objectives=[obj.relevance(rel), obj.diversity(dist)],
        weights=[0.7, 0.3],
        k=10,
        normalize="quantile",
    )
    print("\nreranked top-10 (70% relevance / 30% diversity):", recs.tolist())


if __name__ == "__main__":
    main()
