<div align="center">

# RLProp

**Pick `k` items so your objectives show up in the ratios you ask for** —
e.g. *60% relevance, 30% diversity, 10% novelty* — not just as tuning knobs that
quietly get swallowed by the dominant objective.

<!-- Badges light up once the repo is pushed and the package is published to PyPI. -->
[![PyPI](https://img.shields.io/pypi/v/rlprop-rs.svg)](https://pypi.org/project/rlprop-rs/)
[![CI](https://github.com/pdokoupil/RLProp/actions/workflows/ci.yml/badge.svg)](https://github.com/pdokoupil/RLProp/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Paper](https://img.shields.io/badge/paper-SIGIR'22-b31b1b.svg)](https://doi.org/10.1145/3477495.3531787)

</div>

`RLProp` is *results-level* proportional selection for multi-objective
recommendation. You give it target ratios over objectives and a way to score
items; it greedily builds a list whose **composition** matches those ratios,
using a mandate-allocation / fuzzy-D'Hondt rule borrowed from how parliaments
turn votes into seats. It works for **any** objectives whose per-item marginal
gains you can compute (relevance, diversity, novelty, fairness, calibration, …).

Pure NumPy, one dependency, two functions to learn.

> **Algorithm:** Ladislav Peska & Patrik Dokoupil (SIGIR '22, see [Citing](#citing)).
> **This implementation:** Patrik Dokoupil — a standalone, packaged reimplementation,
> independent of the original experiment code.

## Install

```bash
pip install rlprop-rs
```

## Quick start

Two ways in, depending on what you already have.

> **What's a marginal gain?** RLProp builds the list **incrementally**, one item at
> a time. The *marginal gain* of item `i` for objective `o` is how much adding `i`
> to the current (partial) list `L` improves (or worsens) `o` — naively
> `o(L ∪ {i}) − o(L)`. So `mgains[o, i]` is that quantity for every candidate item
> and every objective. The naive computation re-evaluates `o` for each candidate at
> each step (expensive), but it can often be **avoided or sped up dramatically** —
> e.g. additive objectives like relevance have a constant marginal gain, and many
> objectives admit closed-form incremental updates. See the
> [paper](https://doi.org/10.1145/3477495.3531787) for details and the
> normalization options that make objectives comparable.

**1. You have marginal gains** — an `(n_objectives, n_items)` array. This is the
honest core and the fast path:

```python
import numpy as np
from rlprop import RLProp

mgains = np.random.rand(3, 500)            # gains for 3 objectives over 500 items
top_k = RLProp([0.6, 0.3, 0.1]).select(mgains, k=10)   # -> item indices
```

**2. You have objective functions** — let `rerank` build the marginal gains for
you (it re-scores at every step, so order-dependent objectives like intra-list
diversity work correctly):

```python
from rlprop import rerank
from rlprop import objectives as obj

candidates = np.argsort(-relevance)[:200]  # rerank your model's top-200
recs = rerank(
    candidates,
    objectives=[obj.relevance(relevance), obj.diversity(distance_matrix)],
    weights=[0.7, 0.3],
    k=10,
    normalize="quantile",                  # put objectives on a common scale
)
```

An objective is just a function `f(selected_item_ids) -> float`; the three in
`rlprop.objectives` (relevance, diversity, novelty) are conveniences — writing
your own is one line.

## Two layers, on purpose

The whole public surface is **`RLProp` + `rerank`**. The design is deliberately
flat (no plugin zoo) because over-abstraction kills adoption.

| | `RLProp(weights).select(mgains, k)` | `rerank(candidates, objectives, weights, k)` |
|---|---|---|
| You supply | a marginal-gain matrix | objective value functions |
| Cost | cheap (one pass) | expensive (probes every candidate × objective × step) |
| Best for | additive objectives, precomputed gains, batch jobs | order-dependent objectives, quick experiments |
| Normalization | you do it (or `normalize_mgains`) | built in via `normalize=` |

## What you can tune

- **`weights`** — target ratios, e.g. `[6, 3, 1]` (normalized internally).
- **`allocator`** — `"rlprop"` (default), `"fuzzy_dhondt"` (tie-break toward the
  larger prospective total), or `"weighted_average"` (the non-proportional
  scalarization baseline, for comparison). Or pass your own step function.
- **`normalize`** — `None`, `"minmax"`, `"standard"`, `"robust"`, `"quantile"`
  (all pure NumPy; "objectives on the same scale" is what makes ratios meaningful).
- **`exclude`** — items to never select (already-seen, blocked, etc.).

## How it works

At each step, RLProp looks at how far each objective is from its target share of
the gains awarded so far, and picks the item that best fills the **largest
unfilled mandate** — exactly the D'Hondt highest-averages idea, but over
continuous marginal gains and with support for negative gains. See the
[paper](https://doi.org/10.1145/3477495.3531787) for the derivation and the
relevance–diversity–novelty experiments.

## Relation to the original code & other methods

- The experiments from the SIGIR'22 paper live in the original research
  repository (`moo-as-voting-fast`); this package is the streamlined, install-able
  core extracted from it. The default allocator matches the corrected,
  fully-vectorized form of that code.

## Citing

If you use this software, please cite the paper (GitHub's "Cite this repository"
button reads [`CITATION.cff`](CITATION.cff)):

```bibtex
@inproceedings{peska2022rlprop,
  author    = {Peska, Ladislav and Dokoupil, Patrik},
  title     = {Towards Results-level Proportionality for Multi-objective Recommender Systems},
  booktitle = {Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval},
  series    = {SIGIR '22},
  pages     = {1963--1968},
  year      = {2022},
  doi       = {10.1145/3477495.3531787}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
