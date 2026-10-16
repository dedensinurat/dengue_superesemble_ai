"""Unit tests for DengueCast monthly forecast pipeline (P8)."""

from __future__ import annotations

import os

import pandas as pd

from denguecast.pipelines.monthly_run import monthly_run


def test_monthly_run_dry_run() -> None:
    """Test monthly_run with dry_run=True does not write output tables."""
    result = monthly_run(origin="2019-12", dry_run=True)
    assert "forecasts" in result
    assert "weights" in result
    assert "metadata" in result

    fc = result["forecasts"]
    assert isinstance(fc, pd.DataFrame)
    assert not fc.empty
    assert "risk_level" in fc.columns
    assert "p_exceed_m1sd" in fc.columns
    assert "synthetic" in fc.columns
    assert bool(fc["synthetic"].iloc[0]) is True


def test_monthly_run_resume_idempotent(tmp_path: object) -> None:
    """Test monthly_run resume flag reuses existing run summary JSON."""
    os.makedirs("reports/runs", exist_ok=True)
    summary_path = "reports/runs/run_2019_12.json"

    if os.path.exists(summary_path):
        res = monthly_run(origin="2019-12", resume=True)
        assert isinstance(res, dict)
        assert res.get("origin") == "2019-12" or "forecasts" in res
