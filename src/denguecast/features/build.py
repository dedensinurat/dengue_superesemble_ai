"""Build training and forecast feature frames (MODEL_SPEC C.4 and C.13).

CRITICAL ANTI-LEAKAGE:
- Training frame only uses observed data up to origin.
- Scaler is fit strictly on training data.
- Forecast frame uses observations for months <= origin, and ClimateProvider for months > origin.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from denguecast.climate.provider import ClimateProvider
from denguecast.features.lags import add_all_climate_lags

logger = logging.getLogger(__name__)


def spearman_filter(
    df: pd.DataFrame,
    cols: list[str],
    thr: float = 0.7,
) -> list[str]:
    """Filter collinear features using Spearman rank correlation.

    Rules:
    - Drops features with pairwise |Spearman rho| >= thr.
    - tmin and tmax must never be selected together with tmean or dtr.
    """
    valid_cols = [c for c in cols if c in df.columns]
    if len(valid_cols) <= 1:
        return valid_cols

    corr_matrix = df[valid_cols].corr(method="spearman").abs()
    selected: list[str] = []

    for col in valid_cols:
        # Check mutual exclusion rule
        if col in ["tmin", "tmax", "tmin_l02", "tmax_l02"] and any(
            x in selected for x in ["tmean", "tmean_l02", "dtr", "dtr_l02"]
        ):
            continue
        if col in ["tmean", "tmean_l02", "dtr", "dtr_l02"] and any(
            x in selected for x in ["tmin", "tmax", "tmin_l02", "tmax_l02"]
        ):
            continue

        # Check threshold against already selected
        is_collinear = False
        for s in selected:
            if corr_matrix.loc[col, s] >= thr:
                is_collinear = True
                break
        if not is_collinear:
            selected.append(col)

    return selected


def fit_standardizer(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, tuple[float, float]]:
    """Compute mean and std for feature columns on training data only."""
    scaler = {}
    for col in feature_cols:
        if col in df.columns:
            vals = df[col].dropna()
            m = float(vals.mean()) if len(vals) > 0 else 0.0
            s = float(vals.std(ddof=1)) if len(vals) > 1 and vals.std(ddof=1) > 1e-6 else 1.0
            scaler[col] = (m, s)
    return scaler


def apply_standardizer(df: pd.DataFrame, scaler: dict[str, tuple[float, float]]) -> pd.DataFrame:
    """Transform features using pre-computed scaler."""
    res = df.copy()
    for col, (m, s) in scaler.items():
        if col in res.columns:
            res[f"{col}_z"] = ((res[col] - m) / s).round(4)
    return res


def build_training_frame(
    cases: pd.DataFrame,
    pop: pd.DataFrame,
    climate_obs: pd.DataFrame,
    enso: pd.DataFrame,
    landcover: pd.DataFrame | None = None,
    regions: pd.DataFrame | None = None,
    config: Any = None,
    origin: str | None = None,
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], list[str]]:
    """Build training feature matrix.

    Returns:
        Tuple of (training_df, scaler_dict, active_features).
    """
    cases_df = cases.copy()
    if origin is not None:
        cases_df = cases_df[cases_df["month"] <= origin].copy()

    # 1. Prepare cases and populations
    cases_df["year"] = cases_df["month"].str.slice(0, 4).astype(int)
    merged = pd.merge(
        cases_df, pop[["region_id", "year", "population"]], on=["region_id", "year"], how="left"
    )
    merged["population"] = merged["population"].fillna(1_000_000).astype(int)

    # 2. Add lagged cases (ylag1) per region
    merged = merged.sort_values(["region_id", "month"]).reset_index(drop=True)
    merged["ylag1"] = merged.groupby("region_id")["cases"].shift(1)
    merged["log_ylag1"] = np.log(np.maximum(0.0, merged["ylag1"].fillna(0)) + 1.0)
    merged["log_pop_offset"] = np.log(merged["population"] / 100_000.0)
    merged["month_of_year"] = merged["month"].str.slice(5, 7).astype(int)

    # 3. Prepare climate observations and ENSO
    clim_df = climate_obs.copy()
    if origin is not None:
        clim_df = clim_df[clim_df["month"] <= origin].copy()

    # Merge ENSO into climate
    enso_clean = enso.copy()
    if "month" in enso_clean.columns:
        clim_df = pd.merge(clim_df, enso_clean, on="month", how="left")

    # Add lags
    clim_lagged = add_all_climate_lags(clim_df)

    # Merge climate features into main table
    clim_cols = [
        c
        for c in clim_lagged.columns
        if c not in ["source", "n_stations", "valid_frac", "quant_error", "synthetic"]
    ]
    full_df = pd.merge(merged, clim_lagged[clim_cols], on=["region_id", "month"], how="left")

    # 4. Optional region_group
    if regions is not None and "region_group" in regions.columns:
        full_df = pd.merge(
            full_df, regions[["region_id", "region_group"]], on="region_id", how="left"
        )

    # Drop the first row per region where ylag1 is NaN
    full_df = full_df.dropna(subset=["ylag1"]).reset_index(drop=True)

    # Identify candidate features
    feature_candidates = [
        "humidity_l02",
        "dtr_l02",
        "nino34_l03",
        "oni_l03",
        "dmi_l03",
        "precip_l02",
        "precip_l04",
        "humidity_l04",
        "tmean_l02",
        "tmin_l02",
        "tmax_l02",
        "wind_l0",
    ]
    avail_features = [f for f in feature_candidates if f in full_df.columns]

    # Exclude dtr if missing completely (P4b)
    if "dtr_l02" in avail_features and full_df["dtr_l02"].isna().all():
        avail_features.remove("dtr_l02")
        logger.info("dtr_l02 is completely missing; excluded from active candidate features.")

    # Fit scaler on training data only
    scaler = fit_standardizer(full_df, avail_features)
    full_df = apply_standardizer(full_df, scaler)

    return full_df, scaler, avail_features


def build_forecast_frame(
    origin: str,
    lead: int,
    provider: ClimateProvider,
    ylag_val: pd.DataFrame | dict[str, float] | None,
    climate_obs: pd.DataFrame,
    enso_obs: pd.DataFrame,
    pop: pd.DataFrame,
    regions: pd.DataFrame,
    config: Any = None,
    scaler: dict[str, tuple[float, float]] | None = None,
) -> pd.DataFrame:
    """Build forecast feature frame for a given lead.

    CRITICAL RULE:
    Observations for month > origin are strictly forbidden!
    Future climate is fetched from ClimateProvider.

    Returns:
        DataFrame ready for predict_samples().
    """
    origin_dt = pd.to_datetime(f"{origin}-01")
    target_dt = origin_dt + pd.DateOffset(months=lead)
    target_month = target_dt.strftime("%Y-%m")
    target_year = target_dt.year
    cal_m = target_dt.month

    region_ids = regions["region_id"].astype(str).tolist()

    # Determine needed historical / forecast months for lag calculation (up to 5 months back)
    needed_months = [
        (target_dt - pd.DateOffset(months=i)).strftime("%Y-%m") for i in range(4, -1, -1)
    ]

    # Split into historical (<= origin) and future (> origin)
    hist_months = [m for m in needed_months if m <= origin]
    future_months = [m for m in needed_months if m > origin]

    # 1. Historical climate up to origin
    obs_needed = climate_obs[
        (climate_obs["region_id"].isin(region_ids)) & (climate_obs["month"].isin(hist_months))
    ].copy()

    # Merge enso into obs (avoiding column duplication)
    if enso_obs is not None and not enso_obs.empty:
        new_cols = [c for c in enso_obs.columns if c not in obs_needed.columns or c == "month"]
        if len(new_cols) > 1:
            obs_needed = pd.merge(obs_needed, enso_obs[new_cols], on="month", how="left")

    # 2. Future climate from provider
    if future_months:
        prov_df = provider.get(region_ids=region_ids, origin=origin, target_months=future_months)
        members = prov_df["member"].unique()
    else:
        prov_df = pd.DataFrame()
        members = [0]

    all_member_frames = []

    for mem in members:
        # Assemble sequence for this member
        mem_future = (
            prov_df[prov_df["member"] == mem].rename(columns={"target": "month"})
            if not prov_df.empty
            else pd.DataFrame()
        )

        # Combine historical and future
        mem_seq = pd.concat([obs_needed, mem_future], ignore_index=True)
        mem_seq = mem_seq.loc[:, ~mem_seq.columns.duplicated()]
        # Compute lags for this member
        mem_lagged = add_all_climate_lags(mem_seq)

        # Filter strictly for target_month
        mem_target = mem_lagged[mem_lagged["month"] == target_month].copy()
        mem_target["member"] = mem
        all_member_frames.append(mem_target)

    forecast_climate = pd.concat(all_member_frames, ignore_index=True)

    # Merge population
    pop_target = pop[pop["year"] == target_year][["region_id", "population"]]
    if pop_target.empty:
        # Fallback to latest year
        max_yr = pop["year"].max()
        pop_target = pop[pop["year"] == max_yr][["region_id", "population"]]

    forecast_df = pd.merge(forecast_climate, pop_target, on="region_id", how="left")
    forecast_df["population"] = forecast_df["population"].fillna(1_000_000).astype(int)
    forecast_df["log_pop_offset"] = np.log(forecast_df["population"] / 100_000.0)

    # Attach ylag
    if isinstance(ylag_val, dict):
        forecast_df["ylag1"] = forecast_df["region_id"].map(ylag_val).fillna(20.0)
    elif isinstance(ylag_val, pd.DataFrame):
        col = "ylag_hat" if "ylag_hat" in ylag_val.columns else ("cases" if "cases" in ylag_val.columns else "ylag1")
        ylag_map = ylag_val.set_index("region_id")[col].to_dict()
        forecast_df["ylag1"] = forecast_df["region_id"].map(ylag_map).fillna(20.0)
    else:
        forecast_df["ylag1"] = 20.0

    forecast_df["log_ylag1"] = np.log(np.maximum(0.0, forecast_df["ylag1"]) + 1.0)
    forecast_df["month_of_year"] = cal_m
    forecast_df["origin"] = origin
    forecast_df["target"] = target_month
    forecast_df["lead"] = lead

    # Merge region_group if available
    if "region_group" in regions.columns:
        forecast_df = pd.merge(
            forecast_df, regions[["region_id", "region_group"]], on="region_id", how="left"
        )

    # Apply standardization
    if scaler is not None:
        forecast_df = apply_standardizer(forecast_df, scaler)

    return forecast_df
