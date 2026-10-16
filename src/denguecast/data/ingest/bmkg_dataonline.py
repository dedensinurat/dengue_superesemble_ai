"""BMKG Data Online station adapter (P3a).

Reads daily station files downloaded manually from the BMKG Data Online portal.
Handles codes 8888 (not measured) and 9999 (no data) as NaN — never averaged.
Uses canonical column names only (tmin, tmax, tmean, humidity, precip).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_COLUMN_MAP = {
    "Tanggal": "date",
    "tanggal": "date",
    "DATE": "date",
    "Tn": "tmin",
    "Tx": "tmax",
    "Tavg": "tmean",
    "RH_avg": "humidity",
    "RR": "precip",
    "ss": "sunshine",
    "ff_x": "wind_max",
    "ddd_x": "wind_dir_max",
    "ff_avg": "wind",
    "ddd_car": "wind_dir_mode",
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in km."""
    r = 6371.0  # Earth's radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r * c)


def load_station_file(
    path: str | Path,
    station_id: str,
    config: Any = None,
) -> pd.DataFrame:
    """Load a single BMKG station file (Excel or CSV).

    Args:
        path: Path to station file (.xlsx or .csv).
        station_id: Station identifier.
        config: Optional configuration with bmkg settings.

    Returns:
        Daily canonical DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"BMKG station file not found: {p}")

    # Determine column map, date formats, and special codes
    col_map = DEFAULT_COLUMN_MAP.copy()
    codes = {8888: "nan", 9999: "nan"}
    if config is not None and hasattr(config, "bmkg"):
        bmkg_cfg = config.bmkg
        if hasattr(bmkg_cfg, "column_map") and bmkg_cfg.column_map:
            col_map.update(bmkg_cfg.column_map)
        if hasattr(bmkg_cfg, "codes") and bmkg_cfg.codes:
            codes = bmkg_cfg.codes

    # Read CSV or Excel
    if p.suffix.lower() in [".xlsx", ".xls"]:
        # Find header row
        df_raw = pd.read_excel(p, header=None)
        header_idx = 0
        for i, row in df_raw.head(15).iterrows():
            row_str = " ".join(str(v) for v in row.values)
            if any(k in row_str for k in ["Tanggal", "Tn", "Tx", "Tavg"]):
                header_idx = int(i)
                break
        df = pd.read_excel(p, skiprows=header_idx)
    else:
        # Check first lines for header
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = [f.readline() for _ in range(15)]
        header_idx = 0
        for i, line in enumerate(lines):
            if any(k in line for k in ["Tanggal", "Tn", "Tx", "Tavg"]):
                header_idx = i
                break
        df = pd.read_csv(p, skiprows=header_idx)

    # Rename according to column_map
    rename_dict = {}
    for col in df.columns:
        col_clean = str(col).strip()
        if col_clean in col_map:
            rename_dict[col] = col_map[col_clean]
    df = df.rename(columns=rename_dict)

    if "date" not in df.columns:
        # Try finding a column with dates
        for col in df.columns:
            if "date" in str(col).lower() or "tanggal" in str(col).lower():
                df = df.rename(columns={col: "date"})
                break
        if "date" not in df.columns:
            raise ValueError(f"No date column found in BMKG station file '{p}'")

    # Parse date (BMKG portal uses DD-MM-YYYY as primary format)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["month"] = df["date"].dt.strftime("%Y-%m")
    df["date_str"] = df["date"].dt.strftime("%Y-%m-%d")

    # Replace special codes 8888, 9999 BEFORE any calculations
    numeric_cols = ["tmin", "tmax", "tmean", "humidity", "precip", "sunshine", "wind", "wind_max"]
    present_num = [c for c in numeric_cols if c in df.columns]

    for col in present_num:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        for code, action in codes.items():
            mask = df[col] == code
            if mask.any():
                if action == "zero":
                    df.loc[mask, col] = 0.0
                else:
                    df.loc[mask, col] = np.nan

        # Sanity bounds check
        if col in ["tmin", "tmax", "tmean"]:
            bad_temp = (df[col] < -10.0) | (df[col] > 50.0)
            if bad_temp.any():
                logger.warning(
                    "Replacing %d invalid temperatures in %s with NaN", bad_temp.sum(), col
                )
                df.loc[bad_temp, col] = np.nan
        elif col == "precip":
            bad_rr = df[col] < 0.0
            if bad_rr.any():
                df.loc[bad_rr, col] = np.nan
        elif col == "humidity":
            bad_rh = (df[col] < 0.0) | (df[col] > 100.0)
            if bad_rh.any():
                df.loc[bad_rh, col] = np.nan

    # Calculate daily DTR if tmax and tmin exist
    if "tmax" in df.columns and "tmin" in df.columns:
        df["dtr"] = df["tmax"] - df["tmin"]
        df.loc[df["dtr"] < 0, "dtr"] = np.nan

    df["station_id"] = station_id
    return df


def monthly_from_daily(df: pd.DataFrame, min_valid_frac: float = 0.8) -> pd.DataFrame:
    """Aggregate daily station data to monthly.

    Suhu, kelembaban, angin = rata-rata bulanan.
    Precip = jumlah bulanan.
    DTR = rata-rata harian (tmax - tmin).
    Bulan dengan valid_frac di bawah ambang menjadi NaN.

    Args:
        df: Daily canonical DataFrame from load_station_file.
        min_valid_frac: Minimum valid days fraction (default 0.8).

    Returns:
        Monthly DataFrame with n_valid_days and valid_frac columns.
    """
    rows = []
    station_id = df["station_id"].iloc[0] if "station_id" in df.columns else "UNKNOWN"

    for month_str, group in df.groupby("month"):
        days_in_month = pd.Period(month_str, freq="M").days_in_month

        row: dict[str, Any] = {
            "station_id": station_id,
            "month": month_str,
            "days_in_month": days_in_month,
        }

        # Precip: monthly sum
        if "precip" in group.columns:
            valid_p = group["precip"].dropna()
            n_valid = len(valid_p)
            vfrac = n_valid / float(days_in_month)
            row["precip_n_valid"] = n_valid
            row["precip_valid_frac"] = round(vfrac, 3)
            row["precip"] = float(valid_p.sum()) if vfrac >= min_valid_frac else np.nan

        # Means: tmean, tmin, tmax, humidity, wind, dtr
        mean_vars = ["tmean", "tmin", "tmax", "humidity", "wind", "dtr"]
        for var in mean_vars:
            if var in group.columns:
                valid_vals = group[var].dropna()
                n_valid = len(valid_vals)
                vfrac = n_valid / float(days_in_month)
                row[f"{var}_n_valid"] = n_valid
                row[f"{var}_valid_frac"] = round(vfrac, 3)
                row[var] = float(valid_vals.mean()) if vfrac >= min_valid_frac else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_stations_to_regions(
    monthly: pd.DataFrame,
    stations: pd.DataFrame,
    regions: pd.DataFrame,
    method: str = "idw",
    max_km: float = 150.0,
    min_stations: int = 1,
) -> pd.DataFrame:
    """Aggregate multiple station monthly data to region level.

    Args:
        monthly: Monthly station data [station_id, month, tmean, precip, ...].
        stations: Station metadata [station_id, lat, lon, region_id].
        regions: Region metadata [region_id, lat, lon].
        method: 'idw' (Inverse Distance Weighting), 'mean', or 'nearest'.
        max_km: Maximum radius in km.
        min_stations: Minimum stations required within max_km.

    Returns:
        climate_obs DataFrame with source='bmkg_station'.
    """
    # Merge station coords
    stn_coords = stations.set_index("station_id")[["lat", "lon"]].to_dict("index")
    reg_coords = regions.set_index("region_id")[["lat", "lon"]].to_dict("index")

    all_months = sorted(monthly["month"].unique())
    clim_vars = ["tmean", "tmin", "tmax", "dtr", "precip", "humidity", "wind"]

    output_rows = []

    for rid, rcoord in reg_coords.items():
        rlat, rlon = rcoord["lat"], rcoord["lon"]

        # Calculate distances to all stations
        stn_dists = {}
        for sid, scoord in stn_coords.items():
            d = haversine_km(rlat, rlon, scoord["lat"], scoord["lon"])
            if d <= max_km:
                stn_dists[sid] = d

        for m in all_months:
            m_data = monthly[monthly["month"] == m]
            avail_stns = [s for s in stn_dists if s in m_data["station_id"].values]

            row: dict[str, Any] = {
                "region_id": rid,
                "month": m,
                "source": "bmkg_station",
                "n_stations": len(avail_stns),
                "valid_frac": 1.0 if len(avail_stns) >= min_stations else 0.0,
                "quant_error": 0.0,
            }

            if len(avail_stns) < min_stations:
                for v in clim_vars:
                    row[v] = np.nan
            else:
                subset = m_data[m_data["station_id"].isin(avail_stns)]
                for v in clim_vars:
                    if v not in subset.columns:
                        row[v] = np.nan
                        continue
                    v_valid = subset.dropna(subset=[v])
                    if v_valid.empty:
                        row[v] = np.nan
                        continue

                    if method == "idw":
                        weights = []
                        vals = []
                        for _, srow in v_valid.iterrows():
                            sid = srow["station_id"]
                            dist = stn_dists[sid]
                            w = 1.0 / ((dist + 0.1) ** 2)
                            weights.append(w)
                            vals.append(float(srow[v]))
                        total_w = sum(weights)
                        row[v] = round(
                            float(
                                sum(w * val for w, val in zip(weights, vals, strict=False))
                                / total_w
                            ),
                            2,
                        )
                    elif method == "nearest":
                        # Pick closest station
                        closest_sid = min(v_valid["station_id"], key=lambda sid: stn_dists[sid])
                        row[v] = float(v_valid[v_valid["station_id"] == closest_sid][v].iloc[0])
                    else:  # 'mean'
                        row[v] = round(float(v_valid[v].mean()), 2)

            output_rows.append(row)

    cols = [
        "region_id",
        "month",
        "tmean",
        "tmin",
        "tmax",
        "dtr",
        "precip",
        "humidity",
        "wind",
        "source",
        "n_stations",
        "valid_frac",
        "quant_error",
    ]
    return pd.DataFrame(output_rows, columns=cols)
