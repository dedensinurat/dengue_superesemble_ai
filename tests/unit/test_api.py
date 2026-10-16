"""Unit tests for DengueCast FastAPI service (P9)."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from denguecast.api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    """Check /health returns operational status and clinical disclaimer."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["app"] == "denguecast"
    assert data["version"] == "0.1.0"
    assert "Bukan alat keputusan klinis" in data["disclaimer"]


def test_regions_endpoint() -> None:
    """Check /v1/regions returns list of administrative regions."""
    response = client.get("/v1/regions")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        item = data[0]
        assert "region_id" in item
        assert "name" in item
        assert "region_group" in item
        assert "reliable" in item
        assert isinstance(item["reliable"], bool)


def test_history_endpoint() -> None:
    """Check /v1/history returns time series of historical cases."""
    response = client.get("/v1/history?last_n_months=12")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    if data:
        item = data[0]
        assert "region_id" in item
        assert "month" in item
        assert "cases" in item
        assert "synthetic" in item
        assert isinstance(item["synthetic"], bool)

    # Test filtering by region_id
    if data:
        target_rid = data[0]["region_id"]
        res_filter = client.get(f"/v1/history?region_id={target_rid}&last_n_months=6")
        assert res_filter.status_code == 200
        filtered = res_filter.json()
        assert all(r["region_id"] == target_rid for r in filtered)


def test_evaluation_summary() -> None:
    """Check /v1/evaluation/summary returns aggregated verification metrics."""
    response = client.get("/v1/evaluation/summary")
    assert response.status_code == 200
    data = response.json()
    assert "mean_crps" in data
    assert "median_crpss" in data
    assert "coverage_95" in data
    assert "eligible_provinces" in data
    assert "total_provinces" in data


def test_runs_and_detail_endpoints() -> None:
    """Check /v1/runs listing and /v1/runs/{run_id} retrieval."""
    res_list = client.get("/v1/runs")
    assert res_list.status_code == 200
    runs = res_list.json()
    assert isinstance(runs, list)

    # Non-existent run should 404
    res_404 = client.get("/v1/runs/non_existent_run_999")
    assert res_404.status_code == 404

    # If runs exist, test detail endpoint
    if runs:
        run_id = runs[0]["run_id"]
        res_detail = client.get(f"/v1/runs/{run_id}")
        assert res_detail.status_code == 200
        detail = res_detail.json()
        assert detail["run_id"] == run_id
        assert "origin" in detail
        assert "synthetic" in detail


def test_forecasts_endpoint_with_mock() -> None:
    """Test /v1/forecasts with mock DataFrame."""
    mock_fc = pd.DataFrame(
        [
            {
                "run_id": "test_run_1",
                "region_id": "31",
                "origin": "2020-01",
                "target": "2020-02",
                "lead": 1,
                "mean": 120.5,
                "q025": 80.0,
                "q50": 118.0,
                "q975": 170.0,
                "p_exceed_m1sd": 0.45,
                "p_exceed_m2sd": 0.12,
                "p_exceed_p75": 0.38,
                "p_exceed_p95": 0.08,
                "risk_level": "Waspada",
                "model_version": "0.1.0",
                "synthetic": True,
                "reliable": True,
            }
        ]
    )

    with patch("denguecast.api.main.query_dataframe", return_value=mock_fc):
        res = client.get("/v1/forecasts?lead=1&region_id=31")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        item = data[0]
        assert item["risk_level"] == "Waspada"
        assert item["lead"] == 1
        assert item["mean"] == 120.5
        assert item["reliable"] is True
        assert item["synthetic"] is True


def test_forecasts_map_endpoint_with_mock() -> None:
    """Test /v1/forecasts/map endpoint with threshold mapping."""
    mock_fc = pd.DataFrame(
        [
            {
                "run_id": "test_run_1",
                "region_id": "32",
                "origin": "2020-01",
                "target": "2020-02",
                "lead": 1,
                "mean": 210.0,
                "q50": 205.0,
                "p_exceed_m1sd": 0.65,
                "p_exceed_m2sd": 0.25,
                "p_exceed_p75": 0.50,
                "p_exceed_p95": 0.10,
                "risk_level": "Awas",
                "synthetic": True,
                "reliable": True,
            }
        ]
    )

    with patch("denguecast.api.main.query_dataframe", return_value=mock_fc):
        res = client.get("/v1/forecasts/map?lead=1&threshold=m1sd")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 1
        assert data[0]["region_id"] == "32"
        assert data[0]["p_exceed"] == 0.65
        assert data[0]["risk_level"] == "Awas"


def test_weights_endpoint_with_mock() -> None:
    """Test /v1/weights endpoint."""
    mock_wt = pd.DataFrame(
        [
            {
                "run_id": "test_run_1",
                "model_id": "B1_seasonal_mean",
                "features": "",
                "weight": 0.35,
                "crps": 12.4,
            },
            {
                "run_id": "test_run_1",
                "model_id": "cand_001",
                "features": "rain_sum_lag1_z,temp_mean_lag2_z",
                "weight": 0.65,
                "crps": 11.1,
            },
        ]
    )

    with patch("denguecast.api.main.query_dataframe", return_value=mock_wt):
        res = client.get("/v1/weights?run_id=test_run_1")
        assert res.status_code == 200
        data = res.json()
        assert len(data) == 2
        assert data[0]["model_id"] == "B1_seasonal_mean"
        assert data[1]["features"] == ["rain_sum_lag1_z", "temp_mean_lag2_z"]


def test_download_forecasts_csv_with_mock() -> None:
    """Test /v1/download/forecasts.csv export."""
    mock_fc = pd.DataFrame(
        [
            {
                "run_id": "test_run_1",
                "region_id": "31",
                "origin": "2020-01",
                "target": "2020-02",
                "lead": 1,
                "mean": 120.5,
                "risk_level": "Waspada",
            }
        ]
    )

    with patch("denguecast.api.main.query_dataframe", return_value=mock_fc):
        res = client.get("/v1/download/forecasts.csv?run_id=test_run_1")
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/csv")
        assert "attachment" in res.headers["content-disposition"]
        assert "test_run_1" in res.text
