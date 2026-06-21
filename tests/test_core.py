import numpy as np
import pytest

from rlprop import RLProp
from rlprop.core import _NEG_INF


def test_single_objective_recovers_topk():
    # With all weight on objective 0, RLProp should return its top-k by gain.
    rng = np.random.default_rng(1)
    mgains = rng.random((3, 40))
    top = RLProp([1, 0, 0]).select(mgains, k=8)
    expected = np.argsort(-mgains[0])[:8]
    assert top.tolist() == expected.tolist()


def test_weights_are_normalized():
    assert np.allclose(RLProp([6, 3, 1]).weights, [0.6, 0.3, 0.1])


def test_select_returns_unique_indices_in_range():
    rng = np.random.default_rng(2)
    mgains = rng.random((2, 20))
    top = RLProp([0.5, 0.5]).select(mgains, k=10)
    assert top.size == 10
    assert len(set(top.tolist())) == 10
    assert top.min() >= 0 and top.max() < 20


def test_exclude_items_never_selected():
    rng = np.random.default_rng(3)
    mgains = rng.random((2, 15))
    excluded = {0, 1, 2, 3, 4}
    top = RLProp([0.5, 0.5]).select(mgains, k=10, exclude=excluded)
    assert excluded.isdisjoint(top.tolist())


def test_k_larger_than_available_is_clamped():
    mgains = np.ones((2, 5))
    top = RLProp([0.5, 0.5]).select(mgains, k=100)
    assert top.size == 5


def test_proportionality_beats_weighted_average_on_imbalanced_scale():
    # Objective 1 has a much larger raw scale; a weighted-average baseline gets
    # dominated by it, while RLProp keeps objective 0 represented.
    rng = np.random.default_rng(4)
    mgains = np.stack([rng.random(60), 50.0 * rng.random(60)])
    w = [0.5, 0.5]

    prop = RLProp(w, allocator="rlprop").select(mgains, k=10)
    base = RLProp(w, allocator="weighted_average").select(mgains, k=10)

    def share(sel):
        g = mgains[:, sel].sum(axis=1)
        return g / g.sum()

    # RLProp's objective-0 share is closer to the target 0.5 than the baseline's.
    assert abs(share(prop)[0] - 0.5) < abs(share(base)[0] - 0.5)


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        RLProp([0.5, 0.5]).select(np.ones(10), k=3)        # 1-D mgains
    with pytest.raises(ValueError):
        RLProp([0.5, 0.5]).select(np.ones((3, 10)), k=3)   # objective count mismatch
    with pytest.raises(ValueError):
        RLProp([0, 0])                                     # weights sum to zero


def test_accepts_array_like_inputs():
    # pandas-like / list-of-lists go through np.asarray transparently.
    mgains = [[0.1, 0.9, 0.2], [0.5, 0.1, 0.4]]
    top = RLProp([0.5, 0.5]).select(mgains, k=2)
    assert top.size == 2
