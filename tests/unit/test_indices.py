"""Unit tests for climate indices (ONI and DMI, P3e)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from denguecast.data.ingest.indices import (
    compute_lagged_indices,
    load_dmi,
    load_oni,
    persistence_forecast,
)


def test_parse_csv_indices(tmp_path: Path) -> None:
    oni_csv = tmp_path / "oni.csv"
    dmi_csv = tmp_path / "dmi.csv"

    oni_df = pd.DataFrame(
        {
            "month": ["2020-01", "2020-02", "2020-03", "2020-04"],
            "oni": [0.5, 0.6, 0.7, 0.8],
        }
    )
    oni_df.to_csv(oni_csv, index=False)

    dmi_df = pd.DataFrame(
        {
            "month": ["2020-01", "2020-02", "2020-03", "2020-04"],
            "dmi": [0.1, 0.2, 0.3, 0.4],
        }
    )
    dmi_df.to_csv(dmi_csv, index=False)

    loaded_oni = load_oni(oni_csv)
    loaded_dmi = load_dmi(dmi_csv)

    assert len(loaded_oni) == 4
    assert len(loaded_dmi) == 4
    assert "nino34_anom" in loaded_oni.columns


def test_lagged_indices_rolling_window() -> None:
    """Test 4-month rolling average for oni_l03 and dmi_l03."""
    df = pd.DataFrame(
        {
            "month": ["2020-01", "2020-02", "2020-03", "2020-04"],
            "oni": [1.0, 2.0, 3.0, 4.0],
            "dmi": [0.2, 0.4, 0.6, 0.8],
        }
    )

    res = compute_lagged_indices(df, window=4)
    assert "oni_l03" in res.columns
    assert "dmi_l03" in res.columns

    # 4th month average: (1 + 2 + 3 + 4) / 4 = 2.5
    assert np.isclose(res["oni_l03"].iloc[3], 2.5)
    # 4th month DMI average: (0.2 + 0.4 + 0.6 + 0.8) / 4 = 0.5
    assert np.isclose(res["dmi_l03"].iloc[3], 0.5)


def test_persistence_forecast_anti_leakage() -> None:
    """Persistence forecast must carry forward origin value and NEVER access post-origin data."""
    df_clean = pd.DataFrame(
        {
            "month": ["2020-01", "2020-02", "2020-03"],
            "oni": [0.4, 0.6, 0.8],
        }
    )

    origin = "2020-02"
    fc_clean = persistence_forecast(df_clean, origin=origin, horizon=3, value_col="oni")

    assert len(fc_clean) == 3
    assert fc_clean["target"].tolist() == ["2020-03", "2020-04", "2020-05"]
    assert fc_clean["value"].tolist() == [0.6, 0.6, 0.6]
    assert fc_clean["assumption"].iloc[0] == "persistence"

    # Now corrupt months > origin in the input DataFrame
    df_corrupt = df_clean.copy()
    df_corrupt.loc[df_corrupt["month"] > origin, "oni"] = 999.0

    fc_corrupt = persistence_forecast(df_corrupt, origin=origin, horizon=3, value_col="oni")

    # Forecast must be IDENTICAL
    pd.testing.assert_frame_equal(fc_clean, fc_corrupt)
