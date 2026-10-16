"""Property-based tests for DengueCast evaluation metrics and structures (P11).

Uses hypothesis to test mathematical invariants:
1. CRPS is always non-negative: CRPS(samples, y) >= 0.
2. Degenerate predictive samples equal absolute error: CRPS([c, c, ...], y) == |c - y|.
3. Translation equivariance: CRPS(samples + a, y + a) == CRPS(samples, y).
4. Brier score is bounded between 0 and 1.
5. Superensemble weights are non-negative and sum to 1.0.
6. ExpandingWindowSplitter never leaks future data into training splits.
"""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
import pandas as pd
from hypothesis import given, settings

from denguecast.ensemble.stacking import compute_weights
from denguecast.evaluation.classification import brier_score
from denguecast.evaluation.scores import crps_samples
from denguecast.evaluation.splits import ExpandingWindowSplitter


@settings(max_examples=50, deadline=None)
@given(
    samples=st.lists(
        st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=50,
    ),
    y=st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
)
def test_crps_non_negativity(samples: list[float], y: float) -> None:
    """CRPS must always be greater than or equal to 0."""
    samp_arr = np.array(samples).reshape(1, -1)
    y_arr = np.array([y])
    res = crps_samples(samp_arr, y_arr)
    score = float(np.atleast_1d(res)[0])
    assert score >= -1e-9


@settings(max_examples=50, deadline=None)
@given(
    c=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-500.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    n_samples=st.integers(min_value=5, max_value=30),
)
def test_crps_degenerate_equals_mae(c: float, y: float, n_samples: int) -> None:
    """When all forecast samples equal constant c, CRPS must equal |c - y|."""
    samp_arr = np.full((1, n_samples), c)
    y_arr = np.array([y])
    res = crps_samples(samp_arr, y_arr)
    score = float(np.atleast_1d(res)[0])
    expected = abs(c - y)
    assert np.isclose(score, expected, atol=1e-5)


@settings(max_examples=50, deadline=None)
@given(
    samples=st.lists(
        st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=30,
    ),
    y=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
    shift=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False, allow_infinity=False),
)
def test_crps_translation_equivariance(samples: list[float], y: float, shift: float) -> None:
    """CRPS(samples + a, y + a) == CRPS(samples, y)."""
    samp_arr = np.array(samples).reshape(1, -1)
    y_arr = np.array([y])

    score1 = float(np.atleast_1d(crps_samples(samp_arr, y_arr))[0])
    score2 = float(np.atleast_1d(crps_samples(samp_arr + shift, y_arr + shift))[0])
    assert np.isclose(score1, score2, atol=1e-4)


@settings(max_examples=50, deadline=None)
@given(
    probs=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=5,
        max_size=50,
    ),
    obs=st.lists(st.booleans(), min_size=5, max_size=50),
)
def test_brier_score_bounded(probs: list[float], obs: list[bool]) -> None:
    """Brier score must lie in the unit interval [0, 1]."""
    min_len = min(len(probs), len(obs))
    p_arr = np.array(probs[:min_len])
    o_arr = np.array(obs[:min_len], dtype=int)

    score = brier_score(p_arr, o_arr)
    assert 0.0 <= score <= 1.0


@settings(max_examples=50, deadline=None)
@given(
    scores=st.lists(
        st.floats(min_value=0.01, max_value=1000.0, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=10,
    )
)
def test_weights_sum_to_one_and_positive(scores: list[float]) -> None:
    """Superensemble weights must be positive and sum to 1.0."""
    series = pd.Series(scores, index=[f"M_{i}" for i in range(len(scores))])
    weights = compute_weights(series)

    assert (weights >= 0.0).all()
    assert np.isclose(weights.sum(), 1.0, atol=1e-6)

    # Inversely monotonic: lower CRPS -> higher weight
    if len(scores) >= 2:
        idx_min = series.idxmin()
        idx_max = series.idxmax()
        if series[idx_min] < series[idx_max]:
            assert weights[idx_min] >= weights[idx_max]


def test_splitter_strict_anti_leakage() -> None:
    """ExpandingWindowSplitter must strictly ensure target months > origin and train <= origin."""
    months = [f"201{y}-{m:02d}" for y in range(0, 5) for m in range(1, 13)]
    splitter = ExpandingWindowSplitter(initial_train_months=24, step=6, max_lead=4)
    splits = splitter.split(months)

    assert len(splits) > 0
    for sp in splits:
        # Every train month must be <= origin
        assert all(m <= sp.origin for m in sp.train_months)
        # Every target month must be strictly > origin
        assert all(t > sp.origin for t, _ in sp.test_targets)
        # Leads must be within 1..4
        assert all(1 <= lead <= 4 for _, lead in sp.test_targets)
