"""Lag and rolling-mean feature computation."""

from __future__ import annotations

import pandas as pd


def rolling_mean_lag(
    df: pd.DataFrame,
    var: str,
    window: int = 3,
    shift: int = 0,
    region_col: str = "region_id",
    month_col: str = "month",
) -> pd.Series:
    """Compute rolling mean with lag, strictly per region.

    Never crosses region boundaries.
    Only uses months <= current t (no future data).

    Args:
        df: DataFrame with region_id and month.
        var: Column to compute rolling mean for.
        window: Rolling window size.
        shift: Additional lag shift.
        region_col: Region identifier column name.
        month_col: Month column name.

    Returns:
        Series with rolling mean values aligned with df's original index.
    """
    if var not in df.columns:
        raise KeyError(f"Variable '{var}' not in DataFrame columns.")

    orig_order = df.index
    sorted_df = df.sort_values([region_col, month_col])

    rolled = sorted_df.groupby(region_col)[var].transform(
        lambda s: (s.shift(shift) if shift > 0 else s).rolling(window, min_periods=1).mean()
    )
    return rolled.reindex(orig_order).round(3)


def add_all_climate_lags(
    df: pd.DataFrame,
    climate_window: int = 3,
    nino_window: int = 4,
    include_extended_window: bool = True,
) -> pd.DataFrame:
    """Add all canonical lagged climate features per region.

    Canonical features:
    - tmean_l02, tmin_l02, tmax_l02, dtr_l02, humidity_l02, precip_l02 (window=3)
    - precip_l04, humidity_l04 (extended window=5, for Indonesia patch P4b)
    - wind_l0 (current month t, window=1)
    - nino34_l03, oni_l03, dmi_l03 (window=4)
    """
    out = df.copy()

    # Window 3 (l02: lag 0 to 2)
    for v in ["tmean", "tmin", "tmax", "dtr", "humidity", "precip"]:
        if v in out.columns:
            out[f"{v}_l02"] = rolling_mean_lag(out, v, window=climate_window, shift=0)

    # Window 5 (l04: lag 0 to 4) for Indonesia patch P4b
    if include_extended_window:
        for v in ["precip", "humidity"]:
            if v in out.columns:
                out[f"{v}_l04"] = rolling_mean_lag(out, v, window=5, shift=0)

    # Wind without lag (l0)
    if "wind" in out.columns:
        out["wind_l0"] = rolling_mean_lag(out, "wind", window=1, shift=0)

    # ENSO/IOD windows (window=4)
    for v in ["nino34_anom", "oni", "dmi"]:
        if v in out.columns:
            out[f"{v}_l03"] = rolling_mean_lag(out, v, window=nino_window, shift=0)

    return out
