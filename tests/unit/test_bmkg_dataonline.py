"""Unit tests for BMKG Data Online adapter (P3a)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from denguecast.config import BmkgConfig, DengueCastConfig
from denguecast.data.ingest.bmkg_dataonline import (
    aggregate_stations_to_regions,
    load_station_file,
    monthly_from_daily,
)


@pytest.fixture
def sample_bmkg_csv(tmp_path: Path) -> Path:
    """Create a synthetic BMKG station CSV with 8888, 9999, and Indonesian headers."""
    csv_file = tmp_path / "STN_SLEMAN_daily.csv"
    # 31 days of January 2020
    rows = [
        "Data Iklim Harian Stasiun Sleman",
        "Tanggal,Tn,Tx,Tavg,RH_avg,RR,ff_avg",
    ]
    # Day 1: normal
    rows.append("01-01-2020,22.0,30.0,26.0,80.0,10.0,4.0")
    # Day 2: 8888 in RR (not measured) and 9999 in Tavg (no data)
    rows.append("02-01-2020,23.0,31.0,9999,82.0,8888,5.0")
    # Day 3-31: normal values
    for d in range(3, 32):
        rows.append(f"{d:02d}-01-2020,22.5,30.5,26.5,85.0,5.0,4.5")

    csv_file.write_text("\n".join(rows), encoding="utf-8")
    return csv_file


def test_codes_8888_and_9999_never_averaged(sample_bmkg_csv: Path) -> None:
    """8888 and 9999 must become NaN and never enter means or sums."""
    daily = load_station_file(sample_bmkg_csv, station_id="STN_SLEMAN")

    # In day 2, Tavg was 9999, RR was 8888
    day2 = daily[daily["date_str"] == "2020-01-02"].iloc[0]
    assert pd.isna(day2["tmean"]), "9999 was not converted to NaN!"
    assert pd.isna(day2["precip"]), "8888 was not converted to NaN!"

    # Monthly aggregation
    monthly = monthly_from_daily(daily, min_valid_frac=0.8)
    assert len(monthly) == 1
    m_row = monthly.iloc[0]

    # Mean temperature should be around 26.5, NOT in the thousands!
    assert 20.0 < m_row["tmean"] < 30.0
    # Precip sum should sum only the valid days (10 + 29*5 = 155)
    assert 150.0 <= m_row["precip"] <= 160.0


def test_dtr_and_sums_and_means(sample_bmkg_csv: Path) -> None:
    """Check precip is summed, temperatures averaged, and DTR is tmax - tmin."""
    daily = load_station_file(sample_bmkg_csv, station_id="STN_SLEMAN")
    monthly = monthly_from_daily(daily)
    m = monthly.iloc[0]

    # Daily DTR for day 1 is 30 - 22 = 8
    assert np.isclose(daily.iloc[0]["dtr"], 8.0)
    # Monthly DTR is around 8.0
    assert 7.5 <= m["dtr"] <= 8.5


def test_low_valid_frac_becomes_nan(tmp_path: Path) -> None:
    """If valid days fraction < min_valid_frac, monthly value becomes NaN."""
    csv_file = tmp_path / "sparse.csv"
    rows = ["Tanggal,Tn,Tx,Tavg,RH_avg,RR", "01-01-2020,22,30,26,80,10"]
    # Only 1 valid day out of 31 days in January
    csv_file.write_text("\n".join(rows), encoding="utf-8")

    daily = load_station_file(csv_file, station_id="SPARSE")
    monthly = monthly_from_daily(daily, min_valid_frac=0.8)
    m = monthly.iloc[0]

    assert m["precip_valid_frac"] < 0.1
    assert pd.isna(m["precip"])
    assert pd.isna(m["tmean"])


def test_idw_excludes_stations_beyond_max_km() -> None:
    """Stations beyond max_km must be excluded from IDW regional aggregation."""
    # Monthly data for 2 stations
    monthly = pd.DataFrame(
        [
            {
                "station_id": "STN_NEAR",
                "month": "2020-01",
                "tmean": 25.0,
                "precip": 100.0,
                "tmin": 20.0,
                "tmax": 30.0,
                "dtr": 10.0,
                "humidity": 80.0,
                "wind": 5.0,
            },
            {
                "station_id": "STN_FAR",
                "month": "2020-01",
                "tmean": 35.0,
                "precip": 500.0,
                "tmin": 30.0,
                "tmax": 40.0,
                "dtr": 10.0,
                "humidity": 90.0,
                "wind": 10.0,
            },
        ]
    )

    # Region at (0.0, 0.0)
    regions = pd.DataFrame([{"region_id": "REG_TEST", "lat": 0.0, "lon": 0.0}])

    # STN_NEAR is at (0.1, 0.1) (~15 km), STN_FAR is at (5.0, 5.0) (~780 km)
    stations = pd.DataFrame(
        [
            {"station_id": "STN_NEAR", "lat": 0.1, "lon": 0.1, "region_id": "REG_TEST"},
            {"station_id": "STN_FAR", "lat": 5.0, "lon": 5.0, "region_id": "OTHER"},
        ]
    )

    # max_km = 50km: STN_FAR should be completely ignored
    reg_clim = aggregate_stations_to_regions(
        monthly, stations, regions, method="idw", max_km=50.0, min_stations=1
    )

    assert len(reg_clim) == 1
    row = reg_clim.iloc[0]
    assert row["n_stations"] == 1
    assert np.isclose(row["tmean"], 25.0)
    assert np.isclose(row["precip"], 100.0)


def test_column_map_handles_arbitrary_column_order(tmp_path: Path) -> None:
    """Different column order and custom mapped columns are parsed correctly."""
    csv_file = tmp_path / "custom_cols.csv"
    rows = [
        "Tanggal,Hujan,Suhu_Rata2,Kelembaban",
        "2020-01-01,15.5,27.2,84.0",
        "2020-01-02,0.0,28.1,78.0",
    ]
    csv_file.write_text("\n".join(rows), encoding="utf-8")

    cfg = DengueCastConfig(
        bmkg=BmkgConfig(
            column_map={
                "Tanggal": "date",
                "Hujan": "precip",
                "Suhu_Rata2": "tmean",
                "Kelembaban": "humidity",
            }
        )
    )

    daily = load_station_file(csv_file, station_id="CUSTOM", config=cfg)
    assert "precip" in daily.columns
    assert "tmean" in daily.columns
    assert "humidity" in daily.columns
    assert daily["precip"].iloc[0] == 15.5
