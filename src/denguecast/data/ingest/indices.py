"""Climate indices adapter: ONI and DMI (P3e).

Loads Oceanic Niño Index (ONI) and Dipole Mode Index (DMI).
Computes lagged windows (oni_l03, dmi_l03) and persistence forecasts.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _parse_tabular_or_matrix(path: str | Path, val_col: str) -> pd.DataFrame:
    """Parse either [month, value] CSV or NOAA matrix (YEAR 12-months)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Indices file not found: {p}")

    try:
        df = pd.read_csv(p)
        if "month" in df.columns:
            for c in [val_col, "value", "anom", "oni", "dmi"]:
                if c in df.columns:
                    res = df[["month", c]].rename(columns={c: val_col}).dropna().copy()
                    res["month"] = res["month"].astype(str)
                    res[val_col] = res[val_col].astype(float)
                    return res.sort_values("month").reset_index(drop=True)
    except Exception:
        pass

    # Read matrix format
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 13 and parts[0].isdigit() and len(parts[0]) == 4:
            year = parts[0]
            for m_idx in range(1, 13):
                try:
                    val = float(parts[m_idx])
                    if val < -90.0:  # Missing code
                        continue
                    rows.append({"month": f"{year}-{m_idx:02d}", val_col: val})
                except (ValueError, IndexError):
                    continue

    return pd.DataFrame(rows).sort_values("month").reset_index(drop=True)


def load_oni(path: str | Path) -> pd.DataFrame:
    """Load ONI (Nino 3.4 anomaly) data.

    Returns:
        DataFrame with columns [month, nino34_anom, oni].
    """
    df = _parse_tabular_or_matrix(path, "oni")
    df["nino34_anom"] = df["oni"]
    return df


def load_dmi(path: str | Path) -> pd.DataFrame:
    """Load DMI (Indian Ocean Dipole) data.

    Returns:
        DataFrame with columns [month, dmi_anom, dmi].
    """
    df = _parse_tabular_or_matrix(path, "dmi")
    df["dmi_anom"] = df["dmi"]
    return df


def compute_lagged_indices(df: pd.DataFrame, window: int = 4) -> pd.DataFrame:
    """Compute rolling mean lags for ONI and DMI (months t through t-3)."""
    result = df.sort_values("month").copy()
    if "oni" in result.columns:
        result["oni_l03"] = result["oni"].rolling(window, min_periods=1).mean().round(3)
    if "dmi" in result.columns:
        result["dmi_l03"] = result["dmi"].rolling(window, min_periods=1).mean().round(3)
    return result


def persistence_forecast(
    index_df: pd.DataFrame,
    origin: str,
    horizon: int = 6,
    value_col: str = "oni",
) -> pd.DataFrame:
    """Forecast index by persisting origin-month value into future target months.

    STRICT ANTI-LEAKAGE: Only data with month <= origin is read.

    Args:
        index_df: Historical index DataFrame with columns [month, value_col].
        origin: Forecast origin month (YYYY-MM).
        horizon: Number of lead months (default 6).
        value_col: Column name to persist.

    Returns:
        DataFrame with columns [target, lead, value, assumption].
    """
    hist = index_df[index_df["month"] <= origin].sort_values("month")
    if hist.empty:
        raise ValueError(f"No index observations found on or before origin '{origin}'")

    origin_val = float(hist.iloc[-1][value_col])

    # Generate next target months
    origin_dt = pd.to_datetime(f"{origin}-01")
    target_dates = pd.date_range(
        start=origin_dt + pd.DateOffset(months=1), periods=horizon, freq="MS"
    )

    rows = []
    for h, dt in enumerate(target_dates, start=1):
        rows.append(
            {
                "target": dt.strftime("%Y-%m"),
                "lead": h,
                "value": origin_val,
                "assumption": "persistence",
            }
        )

    return pd.DataFrame(rows)


def load_combined(oni_path: str | Path, dmi_path: str | Path) -> pd.DataFrame:
    """Load both ONI and DMI and compute lagged features."""
    oni_df = load_oni(oni_path)
    dmi_df = load_dmi(dmi_path)

    merged = (
        pd.merge(oni_df, dmi_df, on="month", how="outer")
        .sort_values("month")
        .reset_index(drop=True)
    )
    return compute_lagged_indices(merged)
