"""Unit tests for DengueCast data layer (P3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from denguecast.climate.climatology import ClimatologyProvider
from denguecast.climate.synthetic_ensemble import SyntheticEnsembleProvider
from denguecast.data.schema import (
    validate_cases,
    validate_climate_obs,
    validate_enso,
    validate_landcover,
    validate_population,
)
from denguecast.data.store import compute_df_hash, read_table, write_table
from denguecast.data.synthetic import generate_synthetic_data, get_synthetic_metadata
from denguecast.data.validate import flag_possibly_incomplete, quality_report

# ===========================================================================
# 1. Schema Tests
# ===========================================================================


def test_schema_rejects_negative_cases() -> None:
    bad_df = pd.DataFrame(
        {
            "region_id": ["R01"],
            "month": ["2020-01"],
            "cases": [-5],
            "definition": ["clinical"],
        }
    )
    with pytest.raises(ValueError, match="negative case"):
        validate_cases(bad_df)


def test_schema_rejects_duplicate_cases() -> None:
    bad_df = pd.DataFrame(
        {
            "region_id": ["R01", "R01"],
            "month": ["2020-01", "2020-01"],
            "cases": [10, 20],
            "definition": ["clinical", "clinical"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_cases(bad_df)


def test_schema_rejects_out_of_bounds_temperature() -> None:
    bad_df = pd.DataFrame(
        {
            "region_id": ["R01"],
            "month": ["2020-01"],
            "tmean": [65.0],  # > 50 °C
            "tmin": [20.0],
            "tmax": [25.0],
            "dtr": [5.0],
            "precip": [100.0],
            "humidity": [80.0],
            "wind": [5.0],
        }
    )
    with pytest.raises(ValueError, match="outside bounds"):
        validate_climate_obs(bad_df)


def test_schema_rejects_forbidden_source_column_names() -> None:
    bad_df = pd.DataFrame(
        {
            "region_id": ["R01"],
            "month": ["2020-01"],
            "Tn": [22.0],  # Source name instead of canonical tmin
            "tmean": [26.0],
            "tmin": [22.0],
            "tmax": [30.0],
            "dtr": [8.0],
            "precip": [50.0],
            "humidity": [80.0],
            "wind": [5.0],
        }
    )
    with pytest.raises(ValueError, match="source column names"):
        validate_climate_obs(bad_df)


def test_schema_validates_population_and_enso() -> None:
    pop_df = pd.DataFrame(
        {
            "region_id": ["R01"],
            "year": [2020],
            "population": [500_000],
        }
    )
    validate_population(pop_df)

    enso_df = pd.DataFrame(
        {
            "month": ["2020-01", "2020-02"],
            "nino34_anom": [0.5, 0.4],
        }
    )
    validate_enso(enso_df)

    lc_df = pd.DataFrame(
        {
            "region_id": ["R01"],
            "year": [2020],
            "urban_pct": [30.0],
            "periurban_pct": [20.0],
        }
    )
    validate_landcover(lc_df)


# ===========================================================================
# 2. Synthetic Generator Determinism & Signal Tests
# ===========================================================================


def test_synthetic_data_deterministic() -> None:
    """Same seed must produce bitwise identical data (same hash)."""
    data1 = generate_synthetic_data(seed=123, n_regions=5, n_years=2)
    data2 = generate_synthetic_data(seed=123, n_regions=5, n_years=2)

    for tbl_name in data1:
        hash1 = compute_df_hash(data1[tbl_name])
        hash2 = compute_df_hash(data2[tbl_name])
        assert hash1 == hash2, f"Table {tbl_name} is non-deterministic!"


def test_synthetic_outputs_have_synthetic_flag() -> None:
    data = generate_synthetic_data(seed=42, n_regions=3, n_years=2)
    for tbl_name, df in data.items():
        assert "synthetic" in df.columns, f"Table {tbl_name} missing synthetic flag!"
        assert df["synthetic"].all(), f"Table {tbl_name} synthetic flag is not all True!"


def test_synthetic_metadata_contains_ground_truth() -> None:
    meta = get_synthetic_metadata()
    assert "coefficients" in meta
    assert meta["coefficients"]["humidity_l02"] == 0.35
    assert meta["coefficients"]["dtr_l02"] == -0.25
    assert meta["coefficients"]["nino34_l03"] == 0.20
    assert meta["coefficients"]["wind_l0"] == 0.0


# ===========================================================================
# 3. ClimatologyProvider Anti-Leakage Tests
# ===========================================================================


def test_climatology_never_uses_future_data() -> None:
    """ClimatologyProvider must produce identical output even if future data is corrupted/extreme."""
    # Create base climate observations for 2010-01 through 2020-12
    records = []
    for yr in range(2010, 2021):
        for m in range(1, 13):
            records.append(
                {
                    "region_id": "R01",
                    "month": f"{yr}-{m:02d}",
                    "tmean": 27.0,
                    "tmin": 23.0,
                    "tmax": 31.0,
                    "dtr": 8.0,
                    "precip": 150.0,
                    "humidity": 80.0,
                    "wind": 5.0,
                }
            )
    df_clean = pd.DataFrame(records)

    origin = "2018-06"
    target_months = ["2018-07", "2018-08", "2018-09"]

    provider_clean = ClimatologyProvider(df_clean)
    res_clean = provider_clean.get(["R01"], origin=origin, target_months=target_months)

    # Now corrupt future observations (months > origin) with extreme values
    df_corrupted = df_clean.copy()
    future_mask = df_corrupted["month"] > origin
    df_corrupted.loc[future_mask, "tmean"] = 999.0
    df_corrupted.loc[future_mask, "precip"] = 9999.0
    df_corrupted.loc[future_mask, "humidity"] = 999.0

    provider_corrupted = ClimatologyProvider(df_corrupted)
    res_corrupted = provider_corrupted.get(["R01"], origin=origin, target_months=target_months)

    # The forecasts must be EXACTLY identical because future data was ignored!
    pd.testing.assert_frame_equal(res_clean, res_corrupted)


def test_synthetic_ensemble_provider_shape() -> None:
    df_clim = pd.DataFrame(
        [
            {
                "region_id": "R01",
                "month": "2018-01",
                "tmean": 27.0,
                "tmin": 23.0,
                "tmax": 31.0,
                "dtr": 8.0,
                "precip": 150.0,
                "humidity": 80.0,
                "wind": 5.0,
            }
        ]
    )
    clim_provider = ClimatologyProvider(df_clim)
    ens_provider = SyntheticEnsembleProvider(clim_provider, n_members=20, seed=42)
    ens_df = ens_provider.get(["R01"], origin="2018-01", target_months=["2018-02", "2018-03"])

    # 1 region * 2 target months * 20 members = 40 rows
    assert len(ens_df) == 40
    assert set(ens_df["member"].unique()) == set(range(20))


# ===========================================================================
# 4. Parquet Store and Quality Report Tests
# ===========================================================================


def test_store_write_and_read(tmp_path: object) -> None:
    import pathlib

    p = pathlib.Path(str(tmp_path))
    test_df = pd.DataFrame(
        {
            "region_id": ["R01", "R02"],
            "month": ["2020-01", "2020-01"],
            "cases": [15, 30],
            "definition": ["clinical", "clinical"],
        }
    )

    db_file = p / "test.duckdb"
    h = write_table(test_df, "cases", output_dir=p, db_path=db_file)
    assert len(h) == 64  # sha256 hex length

    loaded = read_table("cases", data_dir=p)
    pd.testing.assert_frame_equal(test_df, loaded)


def test_quality_report_detects_issues() -> None:
    cases = pd.DataFrame(
        {
            "region_id": ["R01", "R01", "R01"],
            "month": ["2020-01", "2020-03", "2020-04"],  # missing 2020-02
            "cases": [10, 15, 20],
            "definition": ["clinical", "clinical", "clinical"],
        }
    )
    climate = pd.DataFrame(
        {
            "region_id": ["R01", "R01", "R01"],
            "month": ["2020-01", "2020-03", "2020-04"],
            "tmean": [26.0, 27.0, 28.0],
            "precip": [100.0, np.nan, 80.0],  # missing precip
        }
    )

    issues_df, summary = quality_report(cases, climate, min_months=12)
    assert not issues_df.empty
    assert "missing_months" in issues_df["issue_type"].values
    assert "missing_values" in issues_df["issue_type"].values
    assert "insufficient_data" in issues_df["issue_type"].values


def test_flag_possibly_incomplete() -> None:
    df = pd.DataFrame({"month": ["2020-01", "2020-02", "2020-03"]})
    flagged = flag_possibly_incomplete(df)
    assert flagged["possibly_incomplete"].tolist() == [False, False, True]
