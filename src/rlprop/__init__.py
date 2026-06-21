"""rlprop — results-level proportionality for multi-objective recommendation.

Pick ``k`` items so the objectives appear in the ratios *you* ask for
(e.g. 60% relevance, 30% diversity, 10% novelty), via mandate-allocation /
fuzzy-D'Hondt selection.

Two ways in:

>>> from rlprop import RLProp
>>> RLProp([0.6, 0.3, 0.1]).select(mgains, k=10)        # you have marginal gains

>>> from rlprop import rerank
>>> from rlprop import objectives as obj
>>> rerank(candidates, [obj.relevance(scores), obj.diversity(dist)],
...        weights=[0.7, 0.3], k=10)                     # you have objective fns

Reference: Peska & Dokoupil, *Towards Results-level Proportionality for
Multi-objective Recommender Systems*, SIGIR '22.
https://doi.org/10.1145/3477495.3531787
"""

from . import objectives
from .core import (
    ALLOCATORS,
    RLProp,
    fuzzy_dhondt_step,
    rlprop_step,
    weighted_average_step,
)
from .normalize import METHODS as NORMALIZATIONS
from .normalize import normalize_mgains
from .rerank import rerank

__all__ = [
    "RLProp",
    "rerank",
    "objectives",
    "ALLOCATORS",
    "NORMALIZATIONS",
    "normalize_mgains",
    "rlprop_step",
    "fuzzy_dhondt_step",
    "weighted_average_step",
]

__version__ = "0.1.0"
