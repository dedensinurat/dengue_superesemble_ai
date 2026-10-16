"""Integration golden regression test for DengueCast demo run (P11).

Verifies that the demo pipeline produces valid, internally consistent
probabilistic forecast outputs across all 63 regions and 6 leads.
"""

from __future__ import annotations

import os

import duckdb
import numpy as np
import pandas as pd
import pytest

from denguecast.pipelines.monthly_run import monthly_run


@pytest.fixture(scope="module")
def demo_run_results() -> dict[str, object]:
    """Execute monthly_run on synthetic demo data or load existing demo run."""
    return monthly_run(origin="2019-12", dry_run=False)


def test_golden_forecast_structure(demo_run_results: dict[str, object]) -> None:
    """Check table dimensions, schema, and required columns."""
    fc = demo_run_results["forecasts"]
    assert isinstance(fc, pd.DataFrame)
    assert not fc.empty

    # 63 regions x 6 leads = 378 rows
    assert len(fc) == 63 * 6
    assert fc["region_id"].nunique() == 63
    assert sorted(fc["lead"].unique().tolist()) == [1, 2, 3, 4, 5, 6]

    required_cols = [
        "run_id",
        "region_id",
        "origin",
        "target",
        "lead",
        "mean",
        "q025",
        "q50",
        "q975",
        "p_exceed_m1sd",
        "p_exceed_m2sd",
        "p_exceed_p75",
        "p_exceed_p95",
        "risk_level",
        "model_version",
        "synthetic",
        "reliable",
    ]
    for col in required_cols:
        assert col in fc.columns, f"Missing column {col} in forecasts"


def test_golden_quantile_monotonicity(demo_run_results: dict[str, object]) -> None:
    """Quantiles must strictly satisfy: q025 <= q50 <= q975."""
    fc = demo_run_results["forecasts"]
    assert isinstance(fc, pd.DataFrame)

    q025 = fc["q025"].to_numpy()
    q50 = fc["q50"].to_numpy()
    q975 = fc["q975"].to_numpy()

    # Allow tiny numerical tolerance
    assert np.all(q025 <= q50 + 1e-5), "Violated q025 <= q50"
    assert np.all(q50 <= q975 + 1e-5), "Violated q50 <= q975"
    assert np.all(q025 >= 0), "Negative case count quantiles observed"


def test_golden_exceedance_probabilities(demo_run_results: dict[str, object]) -> None:
    """Exceedance probabilities must be bounded in [0, 1] and satisfy threshold ordering."""
    fc = demo_run_results["forecasts"]
    assert isinstance(fc, pd.DataFrame)

    p_m1 = fc["p_exceed_m1sd"].to_numpy()
    p_m2 = fc["p_exceed_m2sd"].to_numpy()
    p_p75 = fc["p_exceed_p75"].to_numpy()
    p_p95 = fc["p_exceed_p95"].to_numpy()

    for p, name in [
        (p_m1, "p_exceed_m1sd"),
        (p_m2, "p_exceed_m2sd"),
        (p_p75, "p_exceed_p75"),
        (p_p95, "p_exceed_p95"),
    ]:
        assert np.all(p >= 0.0), f"{name} contains values < 0"
        assert np.all(p <= 1.0), f"{name} contains values > 1"

    # Higher threshold implies lower or equal probability
    assert np.all(p_m2 <= p_m1 + 1e-5), "p_exceed_m2sd > p_exceed_m1sd"
    assert np.all(p_p95 <= p_p75 + 1e-5), "p_exceed_p95 > p_exceed_p75"


def test_golden_risk_levels(demo_run_results: dict[str, object]) -> None:
    """Risk levels must belong to the standard 4 Indonesian categories."""
    fc = demo_run_results["forecasts"]
    assert isinstance(fc, pd.DataFrame)

    valid_levels = {"rendah", "sedang", "tinggi", "sangat_tinggi"}
    levels = set(fc["risk_level"].unique())
    assert levels.issubset(valid_levels), f"Unexpected risk levels: {levels - valid_levels}"


def test_golden_superensemble_weights(demo_run_results: dict[str, object]) -> None:
    """Weights must be non-negative and sum to 1.0 within tolerance."""
    weights = demo_run_results["weights"]
    assert isinstance(weights, pd.DataFrame)
    assert not weights.empty

    assert "weight" in weights.columns
    w = weights["weight"].to_numpy()
    assert np.all(w >= 0.0)
    assert np.isclose(np.sum(w), 1.0, atol=1e-4)


def test_golden_duckdb_persisted() -> None:
    """Verify DuckDB store contains the tables written by the demo run."""
    db_path = "data/processed/denguecast.duckdb"
    if os.path.exists(db_path):
        conn = duckdb.connect(db_path, read_only=True)
        tables = [t[0] for t in conn.execute("SHOW TABLES").fetchall()]
        conn.close()
        for expected in ["forecasts", "ensemble_weights", "province_eligibility"]:
            assert expected in tables, f"Table {expected} not found in DuckDB store"
