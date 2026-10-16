"""Unit tests for feature engineering and epidemic thresholds (P4 and P4b)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from denguecast.climate.climatology import ClimatologyProvider
from denguecast.features.aggregate import aggregate_to_regions
from denguecast.features.build import (
    build_forecast_frame,
    spearman_filter,
)
from denguecast.features.lags import rolling_mean_lag
from denguecast.features.thresholds import endemic_channel, exceedance_prob, risk_level


def test_aggregate_to_regions_weighted() -> None:
    """Test population-weighted spatial aggregation against hand calculation."""
    # Region 1 with 2 cells: cell 1 (weight=100, val=20), cell 2 (weight=300, val=40)
    # Expected weighted mean = (100*20 + 300*40) / 400 = (2000 + 12000) / 400 = 35.0
    cells = pd.DataFrame(
        {
            "region_id": ["R01", "R01"],
            "cell_id": ["C1", "C2"],
            "tmean": [20.0, 40.0],
        }
    )
    weights = pd.DataFrame(
        {
            "region_id": ["R01", "R01"],
            "cell_id": ["C1", "C2"],
            "weight": [100.0, 300.0],
        }
    )

    agg = aggregate_to_regions(cells, weights, val_cols=["tmean"])
    assert len(agg) == 1
    assert np.isclose(agg["tmean"].iloc[0], 35.0)


def test_rolling_lag_never_crosses_regions() -> None:
    """Rolling lag calculation must strictly reset across region boundaries."""
    df = pd.DataFrame(
        {
            "region_id": ["R01", "R01", "R02", "R02"],
            "month": ["2020-01", "2020-02", "2020-01", "2020-02"],
            "val": [10.0, 20.0, 100.0, 200.0],
        }
    )

    lags = rolling_mean_lag(df, var="val", window=2, shift=0)
    # R01 month 1: window=2 on [10] -> 10.0
    assert np.isclose(lags.iloc[0], 10.0)
    # R01 month 2: window=2 on [10, 20] -> 15.0
    assert np.isclose(lags.iloc[1], 15.0)
    # R02 month 1: MUST NOT include R01 values! window=2 on [100] -> 100.0
    assert np.isclose(lags.iloc[2], 100.0)
    # R02 month 2: window=2 on [100, 200] -> 150.0
    assert np.isclose(lags.iloc[3], 150.0)


def test_endemic_channel_anti_leakage() -> None:
    """Endemic channel must only use data up to origin and exclude post-origin data."""
    records = []
    # 2015-01 through 2020-12
    for yr in range(2015, 2021):
        for m in range(1, 13):
            records.append(
                {
                    "region_id": "R01",
                    "month": f"{yr}-{m:02d}",
                    "incidence": 10.0 if yr < 2020 else 500.0,  # massive outbreak in 2020
                }
            )
    df = pd.DataFrame(records)

    origin = "2019-12"
    # Channel with origin 2019-12 must ignore the 2020 outbreak completely
    ch = endemic_channel(df, origin=origin, years=5)
    assert ch["mean"].max() < 20.0


def test_risk_level_and_exceedance_prob() -> None:
    channel = {"mean": 10.0, "m1sd": 15.0, "m2sd": 20.0, "sd": 5.0}

    assert risk_level(5.0, channel) == "rendah"
    assert risk_level(12.0, channel) == "sedang"
    assert risk_level(17.0, channel) == "tinggi"
    assert risk_level(25.0, channel) == "sangat_tinggi"

    samples = np.array([5.0, 12.0, 17.0, 25.0])
    # 2 out of 4 samples exceed 15.0
    assert np.isclose(exceedance_prob(samples, threshold=15.0), 0.5)


def test_spearman_collinearity_filter() -> None:
    """Test spearman filter removes collinear variables and mutual exclusion pairs."""
    x1 = np.linspace(1, 100, 100)
    x2 = x1 + np.random.normal(0, 0.01, 100)  # Almost perfect correlation (rho ~ 1.0)
    x3 = np.random.normal(0, 1, 100)  # Uncorrelated

    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    selected = spearman_filter(df, ["x1", "x2", "x3"], thr=0.7)

    assert "x1" in selected
    assert "x2" not in selected  # Dropped due to collinearity with x1
    assert "x3" in selected

    # Test tmin/tmax vs tmean mutual exclusion
    temp_df = pd.DataFrame(
        {
            "tmean": [25.0, 26.0, 27.0],
            "tmin": [20.0, 21.0, 22.0],
            "tmax": [30.0, 31.0, 32.0],
        }
    )
    sel_temp = spearman_filter(temp_df, ["tmean", "tmin", "tmax"])
    assert "tmean" in sel_temp
    assert "tmin" not in sel_temp
    assert "tmax" not in sel_temp


def test_build_forecast_frame_strictly_no_future_leakage() -> None:
    """CRITICAL TEST: Corrupting observations for month > origin MUST NOT affect forecast features."""
    regions = pd.DataFrame(
        [
            {
                "region_id": "R01",
                "name": "Region 1",
                "lat": 0.0,
                "lon": 100.0,
                "region_group": "Sumatra",
            }
        ]
    )
    pop = pd.DataFrame([{"region_id": "R01", "year": 2020, "population": 1_000_000}])

    # Clean climate observations 2018-01 through 2020-12
    records = []
    for yr in range(2018, 2021):
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
                    "nino34_anom": 0.2,
                    "oni": 0.2,
                    "dmi": 0.1,
                }
            )
    df_clean = pd.DataFrame(records)
    enso_clean = df_clean[["month", "nino34_anom", "oni", "dmi"]].drop_duplicates()

    origin = "2020-05"
    lead = 2  # target month 2020-07

    provider = ClimatologyProvider(df_clean, enso_obs=enso_clean)

    fc_clean = build_forecast_frame(
        origin=origin,
        lead=lead,
        provider=provider,
        ylag_val={"R01": 25.0},
        climate_obs=df_clean,
        enso_obs=enso_clean,
        pop=pop,
        regions=regions,
    )

    # Now corrupt observations > origin with massive anomalies
    df_corrupted = df_clean.copy()
    future_mask = df_corrupted["month"] > origin
    df_corrupted.loc[future_mask, "tmean"] = 999.0
    df_corrupted.loc[future_mask, "precip"] = 99999.0
    df_corrupted.loc[future_mask, "humidity"] = 999.0

    enso_corrupted = enso_clean.copy()
    enso_corrupted.loc[enso_corrupted["month"] > origin, "oni"] = 99.0

    # Provider and forecast frame with corrupted future observations
    provider_corrupted = ClimatologyProvider(df_corrupted, enso_obs=enso_corrupted)
    fc_corrupted = build_forecast_frame(
        origin=origin,
        lead=lead,
        provider=provider_corrupted,
        ylag_val={"R01": 25.0},
        climate_obs=df_corrupted,
        enso_obs=enso_corrupted,
        pop=pop,
        regions=regions,
    )

    # Values for precip, tmean, humidity, ylag must be IDENTICAL!
    for col in ["precip_l02", "tmean_l02", "humidity_l02", "ylag1"]:
        assert np.isclose(
            fc_clean[col].iloc[0], fc_corrupted[col].iloc[0]
        ), f"Leakage detected in {col}!"
