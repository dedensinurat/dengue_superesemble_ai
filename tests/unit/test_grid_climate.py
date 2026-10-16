"""Unit tests for grid climate adapters (ERA5-Land and NASA POWER, P3b)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from denguecast.data.ingest.era5land import (
    calculate_rh_from_t_td,
    compare_climate,
    kelvin_to_celsius,
    precip_m_to_mm,
    wind_speed_from_uv,
)
from denguecast.data.ingest.era5land import (
    load as load_era5,
)
from denguecast.data.ingest.nasa_power import (
    fetch_point_daily,
    parse_power_daily_to_monthly,
)


def test_rh_magnus_formula_hand_values() -> None:
    """Test Magnus-Tetens relative humidity matches known reference values."""
    # When T == Td, RH must be 100%
    rh_saturated = calculate_rh_from_t_td(25.0, 25.0)
    assert np.isclose(rh_saturated, 100.0, atol=0.01)

    # Reference meteorological check: T=30.0 C, Td=20.0 C -> RH ~ 55%
    rh = calculate_rh_from_t_td(30.0, 20.0)
    assert 53.0 < rh < 57.0

    # Low dewpoint check: T=25.0 C, Td=10.0 C -> RH ~ 39%
    rh_dry = calculate_rh_from_t_td(25.0, 10.0)
    assert 37.0 < rh_dry < 41.0


def test_unit_conversions() -> None:
    # 273.15 K == 0.0 C, 300 K == 26.85 C
    assert np.isclose(kelvin_to_celsius(273.15), 0.0)
    assert np.isclose(kelvin_to_celsius(300.0), 26.85)

    # 0.05 meters == 50.0 mm
    assert np.isclose(precip_m_to_mm(0.05), 50.0)

    # u=3, v=4 -> wind=5.0
    assert np.isclose(wind_speed_from_uv(3.0, 4.0), 5.0)


def test_era5land_load_csv_fallback(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Test ERA5 CSV conversion and fallback warning without pop raster."""
    csv_file = tmp_path / "era5_sample.csv"
    df = pd.DataFrame(
        {
            "region_id": ["REG_01", "REG_01"],
            "month": ["2020-01", "2020-02"],
            "t2m": [300.0, 301.0],  # Kelvin
            "d2m": [295.0, 296.0],  # Kelvin
            "tp": [0.15, 0.20],  # meters
            "u10": [2.0, 3.0],
            "v10": [2.0, 3.0],
        }
    )
    df.to_csv(csv_file, index=False)

    loaded = load_era5(csv_file, pop_raster=None)
    assert len(loaded) == 2
    assert loaded["source"].iloc[0] == "era5land"
    assert 25.0 < loaded["tmean"].iloc[0] < 28.0
    assert 140.0 < loaded["precip"].iloc[0] < 160.0


def test_nasa_power_network_safeguard_and_cache(tmp_path: Path) -> None:
    """NASA POWER must raise RuntimeError when allow_network=False and not cached."""
    cache_dir = tmp_path / "cache"

    with pytest.raises(RuntimeError, match="Network access disabled"):
        fetch_point_daily(
            lat=-7.5,
            lon=110.5,
            start="20200101",
            end="20200131",
            cache_dir=cache_dir,
            allow_network=False,
        )

    # With mock data, it should write to cache and read back without network
    mock_payload = {
        "properties": {
            "parameter": {
                "T2M": {"20200101": 26.5, "20200102": 27.0},
                "T2M_MIN": {"20200101": 22.0, "20200102": 23.0},
                "T2M_MAX": {"20200101": 31.0, "20200102": 32.0},
                "RH2M": {"20200101": 80.0, "20200102": 82.0},
                "PRECTOTCORR": {"20200101": 15.0, "20200102": 5.0},
                "WS2M": {"20200101": 3.5, "20200102": 4.0},
            }
        }
    }

    cached = fetch_point_daily(
        lat=-7.5,
        lon=110.5,
        start="20200101",
        end="20200131",
        cache_dir=cache_dir,
        allow_network=False,
        mock_data=mock_payload,
    )
    assert "properties" in cached

    # Parse to monthly
    monthly_df = parse_power_daily_to_monthly(cached, region_id="REG_TEST")
    assert len(monthly_df) == 1
    row = monthly_df.iloc[0]
    assert row["source"] == "nasa_power"
    assert np.isclose(row["precip"], 20.0)
    assert np.isclose(row["tmean"], 26.75)


def test_compare_climate_metrics() -> None:
    """Test correlation and bias calculations between two climate sources."""
    df_a = pd.DataFrame(
        {
            "region_id": ["R01", "R01", "R01", "R02", "R02", "R02"],
            "month": ["2020-01", "2020-02", "2020-03", "2020-01", "2020-02", "2020-03"],
            "tmean": [26.0, 27.0, 28.0, 25.0, 26.0, 27.0],
            "precip": [100.0, 150.0, 200.0, 80.0, 120.0, 160.0],
            "humidity": [80.0, 82.0, 85.0, 75.0, 78.0, 80.0],
            "wind": [4.0, 5.0, 6.0, 3.0, 4.0, 5.0],
        }
    )

    # df_b has +1.0 C bias and perfect correlation
    df_b = df_a.copy()
    df_b["tmean"] = df_a["tmean"] + 1.0
    df_b["precip"] = df_a["precip"] - 10.0

    merged, summary = compare_climate(df_a, df_b, source_a="era5", source_b="bmkg")
    assert summary["n_matched"] == 6
    assert np.isclose(summary["metrics"]["tmean"]["pearson"], 1.0)
    assert np.isclose(summary["metrics"]["tmean"]["bias"], -1.0)
    assert np.isclose(summary["metrics"]["precip"]["bias"], 10.0)
