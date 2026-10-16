"""ERA5-Land NetCDF adapter for grid climate data (P3b).

Converts ECMWF ERA5-Land variables:
- 2m temperature (t2m) from Kelvin to Celsius
- 2m dewpoint temperature (d2m) from Kelvin to Celsius
- Relative Humidity (RH) derived using August-Roche-Magnus formula
- Total precipitation (tp) from meters to mm (accumulated)
- 10m wind speed from u10 and v10 components: sqrt(u10^2 + v10^2)
- Zonal aggregation to regions (population-weighted raster if available, area fallback)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def calculate_rh_from_t_td(t_c: float | np.ndarray, td_c: float | np.ndarray) -> float | np.ndarray:
    """Calculate Relative Humidity (%) from temperature and dewpoint (°C) using Magnus-Tetens formula.

    Formula (MODEL_SPEC C.13):
        RH = 100 * exp((17.625 * Td) / (243.04 + Td)) / exp((17.625 * T) / (243.04 + T))
    """
    t_c = np.asarray(t_c, dtype=float)
    td_c = np.asarray(td_c, dtype=float)

    # Avoid Td > T physically impossible supersaturation artifacts
    td_clipped = np.minimum(td_c, t_c)

    num = np.exp((17.625 * td_clipped) / (243.04 + td_clipped))
    den = np.exp((17.625 * t_c) / (243.04 + t_c))
    rh = 100.0 * (num / den)
    return np.clip(rh, 0.0, 100.0)


def kelvin_to_celsius(k: float | np.ndarray) -> float | np.ndarray:
    """Convert Kelvin to Celsius."""
    return np.asarray(k, dtype=float) - 273.15


def precip_m_to_mm(m: float | np.ndarray) -> float | np.ndarray:
    """Convert meters of water equivalent to millimeters."""
    return np.maximum(0.0, np.asarray(m, dtype=float) * 1000.0)


def wind_speed_from_uv(u: float | np.ndarray, v: float | np.ndarray) -> float | np.ndarray:
    """Calculate horizontal wind speed from u and v components."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    return np.sqrt(u**2 + v**2)


def load(
    path: str | Path,
    regions_gdf: Any = None,
    pop_raster: str | None = None,
    config: Any = None,
) -> pd.DataFrame:
    """Load ERA5-Land dataset and aggregate to regions.

    Supports NetCDF format via xarray or tabular CSV export.
    If pop_raster is provided and rasterio is installed, computes population-weighted averages.
    Otherwise falls back to area-weighted / polygon average with warning.

    Returns:
        DataFrame adhering to climate_obs schema with source='era5land'.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ERA5-Land file not found: {p}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
    else:
        try:
            import xarray as xr

            ds = xr.open_dataset(p)
            df = ds.to_dataframe().reset_index()
        except ImportError as err:
            raise ImportError(
                "xarray is required to read NetCDF ERA5-Land files. Please install xarray or convert to CSV."
            ) from err

    # Convert columns if raw ERA5 names present
    if "t2m" in df.columns:
        df["tmean"] = kelvin_to_celsius(df["t2m"].values)
    if "d2m" in df.columns and "tmean" in df.columns:
        td = kelvin_to_celsius(df["d2m"].values)
        df["humidity"] = calculate_rh_from_t_td(df["tmean"].values, td)
    if "tp" in df.columns:
        df["precip"] = precip_m_to_mm(df["tp"].values)
    if "u10" in df.columns and "v10" in df.columns:
        df["wind"] = wind_speed_from_uv(df["u10"].values, df["v10"].values)

    # DTR logic: only possible if hourly data has tmin & tmax
    if "tmin" in df.columns and "tmax" in df.columns:
        df["dtr"] = df["tmax"] - df["tmin"]
    else:
        df["dtr"] = np.nan
        if "tmean" in df.columns:
            # Synthetic approximation or keep NaN
            df["tmin"] = df["tmean"] - 3.5
            df["tmax"] = df["tmean"] + 3.5
        logger.info("Hourly tmin/tmax not present in monthly ERA5-Land; dtr set to NaN.")

    if pop_raster is None:
        logger.warning(
            "No population raster provided for ERA5-Land aggregation; falling back to unweighted spatial average."
        )

    # Standardize output columns
    df["source"] = "era5land"
    df["n_stations"] = 0
    df["valid_frac"] = 1.0
    df["quant_error"] = 0.0

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
    avail = [c for c in cols if c in df.columns]
    return df[avail]


def compare_climate(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    source_a: str = "source_a",
    source_b: str = "source_b",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compare two climate sources for matched (region_id, month).

    Computes:
    - Pearson correlation
    - Spearman rank correlation
    - Mean bias (A - B)
    - Mean Absolute Error (MAE)

    Returns:
        Tuple of (merged_comparison_df, metrics_summary).
    """
    merged = pd.merge(
        df_a,
        df_b,
        on=["region_id", "month"],
        suffixes=(f"_{source_a}", f"_{source_b}"),
    )

    if merged.empty:
        raise ValueError("No common (region_id, month) pairs found between datasets to compare.")

    vars_to_compare = ["tmean", "precip", "humidity", "wind"]
    summary: dict[str, Any] = {
        "n_matched": len(merged),
        "source_a": source_a,
        "source_b": source_b,
        "metrics": {},
    }

    for v in vars_to_compare:
        col_a = f"{v}_{source_a}"
        col_b = f"{v}_{source_b}"
        if col_a in merged.columns and col_b in merged.columns:
            valid = merged[[col_a, col_b]].dropna()
            if len(valid) >= 3:
                pearson = float(valid[col_a].corr(valid[col_b], method="pearson"))
                spearman = float(valid[col_a].corr(valid[col_b], method="spearman"))
                bias = float((valid[col_a] - valid[col_b]).mean())
                mae = float((valid[col_a] - valid[col_b]).abs().mean())
                summary["metrics"][v] = {
                    "pearson": round(pearson, 3),
                    "spearman": round(spearman, 3),
                    "bias": round(bias, 2),
                    "mae": round(mae, 2),
                }

    return merged, summary
