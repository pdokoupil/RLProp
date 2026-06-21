import numpy as np
import pytest

from rlprop import rerank
from rlprop import objectives as obj
from rlprop.normalize import METHODS, normalize_mgains


def test_relevance_only_is_greedy_topk():
    rng = np.random.default_rng(11)
    rel = rng.random(40)
    candidates = np.arange(40)
    recs = rerank(candidates, [obj.relevance(rel)], weights=[1.0], k=10)
    assert recs.tolist() == np.argsort(-rel)[:10].tolist()


def test_rerank_respects_candidate_pool_and_exclude():
    rng = np.random.default_rng(12)
    rel = rng.random(40)
    candidates = np.argsort(-rel)[:20]
    recs = rerank(
        candidates, [obj.relevance(rel)], weights=[1.0], k=5, exclude={int(candidates[0])}
    )
    assert int(candidates[0]) not in recs.tolist()
    assert set(recs.tolist()).issubset(set(candidates.tolist()))


def test_diversity_changes_selection():
    rng = np.random.default_rng(13)
    rel = rng.random(30)
    dist = rng.random((30, 30))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0.0)
    candidates = np.arange(30)

    pure_rel = rerank(candidates, [obj.relevance(rel), obj.diversity(dist)],
                      weights=[1.0, 0.0], k=10, normalize="quantile")
    with_div = rerank(candidates, [obj.relevance(rel), obj.diversity(dist)],
                      weights=[0.5, 0.5], k=10, normalize="quantile")
    assert pure_rel.tolist() != with_div.tolist()


@pytest.mark.parametrize("method", METHODS)
def test_normalizers_run_and_are_finite(method):
    rng = np.random.default_rng(14)
    mgains = rng.normal(size=(3, 25))
    out = normalize_mgains(mgains, method)
    assert out.shape == mgains.shape
    assert np.all(np.isfinite(out))


def test_quantile_maps_to_unit_interval():
    rng = np.random.default_rng(15)
    mgains = rng.normal(size=(2, 50))
    out = normalize_mgains(mgains, "quantile")
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_unknown_objective_count_raises():
    with pytest.raises(ValueError):
        rerank(np.arange(10), [obj.relevance(np.ones(10))], weights=[0.5, 0.5], k=3)
