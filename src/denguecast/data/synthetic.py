"""Synthetic data generator for DengueCast.

Generates reproducible synthetic data for testing and demo purposes.
All outputs are labelled synthetic=True.

Generator creates:
- 63 fictitious regions (7×9 grid with neighbor list)
- 12 years of monthly data
- Climate: seasonal (sine) + AR(1) noise + regional gradient
- Cases: NB with known ground-truth coefficients
- One outbreak year anomaly + some missing values

Ground-truth coefficients (stored in metadata):
    humidity_l02: +0.35
    dtr_l02:      -0.25
    nino34_l03:   +0.20
    wind_l0:       0.00  (no effect)
    rho:          +0.40  (lagged case effect)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from denguecast.data.store import write_table

# Ground-truth coefficients for synthetic data
SYNTHETIC_COEFS: dict[str, float] = {
    "humidity_l02": 0.35,
    "dtr_l02": -0.25,
    "nino34_l03": 0.20,
    "wind_l0": 0.0,
    "rho": 0.4,  # AR coefficient for lagged cases
}

N_REGIONS: int = 63  # 7 × 9 grid
N_YEARS: int = 12
START_YEAR: int = 2010


def generate_synthetic_regions(n_regions: int = N_REGIONS) -> pd.DataFrame:
    """Generate 7x9 grid of regions with coordinates and region_groups."""
    rows = []
    groups = ["Sumatra", "Java-Bali", "Kalimantan", "Sulawesi", "Nusa Tenggara", "Maluku", "Papua"]

    # 7 rows x 9 cols grid
    n_rows = 7
    n_cols = int(np.ceil(n_regions / n_rows))

    for idx in range(n_regions):
        r = idx // n_cols
        c = idx % n_cols
        rid = f"REG_{idx+1:02d}"
        group = groups[r % len(groups)]
        # Fictitious coordinates in Indonesia bounding box [-10 to 5 lat, 95 to 141 lon]
        lat = -8.0 + r * 2.0 + (c % 2) * 0.2
        lon = 96.0 + c * 5.0 + (r % 2) * 0.2
        rows.append(
            {
                "region_id": rid,
                "name": f"Region {idx+1}",
                "region_group": group,
                "grid_row": r,
                "grid_col": c,
                "lat": round(lat, 4),
                "lon": round(lon, 4),
                "synthetic": True,
            }
        )
    return pd.DataFrame(rows)


def generate_synthetic_data(
    seed: int = 42,
    n_regions: int = N_REGIONS,
    n_years: int = N_YEARS,
    output_dir: str | Path | None = None,
    db_path: str | Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate all synthetic tables deterministically.

    Args:
        seed: Random seed for full reproducibility.
        n_regions: Number of synthetic regions.
        n_years: Number of years of monthly data.
        output_dir: If provided, write Parquet files here.
        db_path: Path to DuckDB file for registration.

    Returns:
        Dict mapping table name to DataFrame, all with synthetic=True column.
    """
    rng = np.random.default_rng(seed)

    # 1. Regions
    regions_df = generate_synthetic_regions(n_regions)
    region_ids = regions_df["region_id"].tolist()

    # 2. Timeline
    months = []
    years = []
    for yr in range(START_YEAR, START_YEAR + n_years):
        for cal_m in range(1, 13):
            months.append(f"{yr}-{cal_m:02d}")
            years.append(yr)
    n_months = len(months)
    unique_years = sorted(list(set(years)))

    # 3. Population table
    pop_rows = []
    base_pops = rng.lognormal(mean=13.8, sigma=0.6, size=n_regions).astype(int)  # ~1M population
    for idx, rid in enumerate(region_ids):
        base_pop = base_pops[idx]
        for yr_idx, yr in enumerate(unique_years):
            # 1.2% annual growth
            cur_pop = int(base_pop * ((1.0 + 0.012) ** yr_idx))
            pop_rows.append(
                {
                    "region_id": rid,
                    "year": yr,
                    "population": cur_pop,
                    "synthetic": True,
                }
            )
    population_df = pd.DataFrame(pop_rows)
    pop_lookup = population_df.set_index(["region_id", "year"])["population"].to_dict()

    # 4. ENSO table
    # AR(1) process with ~4 year period
    enso_noise = rng.normal(0, 0.35, n_months)
    nino_vals = np.zeros(n_months)
    nino_vals[0] = enso_noise[0]
    for t in range(1, n_months):
        nino_vals[t] = 0.82 * nino_vals[t - 1] + enso_noise[t]

    # Outbreak year: Year 2016 (months 72-83) has elevated El Nino
    outbreak_year = START_YEAR + 6
    for t, m in enumerate(months):
        if m.startswith(str(outbreak_year)):
            nino_vals[t] += 1.2

    enso_rows = []
    for t, m in enumerate(months):
        enso_rows.append(
            {
                "month": m,
                "nino34_anom": round(float(nino_vals[t]), 3),
                "oni": round(float(nino_vals[t]), 3),
                "dmi": round(float(nino_vals[t] * 0.4 + rng.normal(0, 0.2)), 3),
                "synthetic": True,
            }
        )
    enso_df = pd.DataFrame(enso_rows)
    nino_lookup = enso_df.set_index("month")["nino34_anom"].to_dict()

    # 5. Landcover table
    lc_rows = []
    for rid in region_ids:
        base_urban = rng.uniform(10.0, 60.0)
        base_peri = rng.uniform(15.0, 35.0)
        for yr_idx, yr in enumerate(unique_years):
            urban = min(95.0, base_urban + yr_idx * 0.5)
            peri = max(5.0, base_peri - yr_idx * 0.2)
            lc_rows.append(
                {
                    "region_id": rid,
                    "year": yr,
                    "urban_pct": round(urban, 2),
                    "periurban_pct": round(peri, 2),
                    "synthetic": True,
                }
            )
    landcover_df = pd.DataFrame(lc_rows)

    # 6. Climate observations
    clim_rows = []
    # Regional baselines
    reg_temp_base = 26.5 + rng.normal(0, 0.8, n_regions)
    reg_precip_base = 180.0 + rng.normal(0, 30.0, n_regions)

    for i, rid in enumerate(region_ids):
        t_base = reg_temp_base[i]
        p_base = max(80.0, reg_precip_base[i])

        # AR(1) climate noise
        t_noise = rng.normal(0, 0.4, n_months)
        p_noise = rng.normal(0, 25.0, n_months)
        h_noise = rng.normal(0, 3.0, n_months)

        for t, m in enumerate(months):
            cal_month = int(m.split("-")[1])
            # Seasonality: wet season Dec-Mar (peak month 1), dry season Jul-Aug (trough month 7)
            phase = 2 * np.pi * (cal_month - 1) / 12.0

            # Rainfall: high in wet season
            precip = max(5.0, p_base + 100.0 * np.cos(phase) + p_noise[t])
            # Temperature: slightly cooler in wet season, warmer in dry
            tmean = t_base - 0.8 * np.cos(phase) + t_noise[t]
            # DTR (diurnal temp range): higher in dry season
            dtr = max(3.0, 7.5 - 2.5 * np.cos(phase) + rng.normal(0, 0.4))
            tmin = tmean - dtr / 2.0
            tmax = tmean + dtr / 2.0
            # Humidity: higher with rain
            humidity = float(np.clip(72.0 + 15.0 * np.cos(phase) + h_noise[t], 45.0, 98.0))
            # Wind: 3 to 12 km/h
            wind = max(0.5, 5.0 + 2.0 * np.sin(phase) + rng.exponential(1.2))

            clim_rows.append(
                {
                    "region_id": rid,
                    "month": m,
                    "tmean": round(float(tmean), 2),
                    "tmin": round(float(tmin), 2),
                    "tmax": round(float(tmax), 2),
                    "dtr": round(float(dtr), 2),
                    "precip": round(float(precip), 1),
                    "humidity": round(float(humidity), 1),
                    "wind": round(float(wind), 2),
                    "source": "synthetic",
                    "n_stations": 5,
                    "valid_frac": 1.0,
                    "quant_error": 0.0,
                    "synthetic": True,
                }
            )
    climate_obs_df = pd.DataFrame(clim_rows)

    # Insert a few isolated missing values in climate_obs to test validate reporting
    nan_indices = rng.choice(len(climate_obs_df), size=8, replace=False)
    climate_obs_df.loc[nan_indices, "wind"] = np.nan

    # 7. Cases generation using ground truth coefficients
    # Sort for rolling computations
    climate_sorted = climate_obs_df.sort_values(["region_id", "month"]).copy()

    # Compute rolling values per region
    climate_sorted["humidity_l02"] = climate_sorted.groupby("region_id")["humidity"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    climate_sorted["dtr_l02"] = climate_sorted.groupby("region_id")["dtr"].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    climate_sorted["nino34"] = climate_sorted["month"].map(nino_lookup)
    climate_sorted["nino34_l03"] = climate_sorted.groupby("region_id")["nino34"].transform(
        lambda s: s.rolling(4, min_periods=1).mean()
    )

    # Standardize predictors (z-score) across the dataset for clean coefficient simulation
    h_mean, h_std = climate_sorted["humidity_l02"].mean(), climate_sorted["humidity_l02"].std()
    d_mean, d_std = climate_sorted["dtr_l02"].mean(), climate_sorted["dtr_l02"].std()
    n_mean, n_std = climate_sorted["nino34_l03"].mean(), climate_sorted["nino34_l03"].std()

    climate_sorted["h_z"] = (climate_sorted["humidity_l02"] - h_mean) / h_std
    climate_sorted["d_z"] = (climate_sorted["dtr_l02"] - d_mean) / d_std
    climate_sorted["n_z"] = (climate_sorted["nino34_l03"] - n_mean) / n_std

    clim_dict = climate_sorted.set_index(["region_id", "month"])[["h_z", "d_z", "n_z"]].to_dict(
        "index"
    )

    # Fixed effects
    region_effects = rng.normal(0.0, 0.3, n_regions)
    reg_eff_map = {rid: region_effects[i] for i, rid in enumerate(region_ids)}

    case_rows = []
    alpha_dispersion = 0.25  # Negative binomial dispersion parameter

    for rid in region_ids:
        a_i = reg_eff_map[rid]
        prev_y = 20  # initial lagged cases

        for _t, m in enumerate(months):
            yr = int(m.split("-")[0])
            cal_month = int(m.split("-")[1])
            pop = pop_lookup[(rid, yr)]
            pop_offset = np.log(pop / 100_000.0)

            # Seasonal month effect
            season_eff = 0.4 * np.cos(2 * np.pi * (cal_month - 2) / 12.0)

            # Predictors
            features = clim_dict.get((rid, m), {"h_z": 0.0, "d_z": 0.0, "n_z": 0.0})

            # log mu formula matching MODEL_SPEC C.3
            log_mu = (
                a_i
                + 3.2
                + pop_offset
                + season_eff
                + SYNTHETIC_COEFS["rho"] * np.log(prev_y + 1)
                + SYNTHETIC_COEFS["humidity_l02"] * features["h_z"]
                + SYNTHETIC_COEFS["dtr_l02"] * features["d_z"]
                + SYNTHETIC_COEFS["nino34_l03"] * features["n_z"]
                + SYNTHETIC_COEFS["wind_l0"] * 0.0
            )

            # Outbreak anomaly boost in 2016
            if yr == outbreak_year:
                log_mu += 0.65

            mu = np.exp(np.clip(log_mu, 0.0, 9.5))

            # Sample from Negative Binomial with mean mu and variance mu + alpha*mu^2
            # scipy nbinom uses n (number of successes) and p (success probability):
            # mean = n*(1-p)/p = mu; var = n*(1-p)/p^2 = mu + alpha*mu^2
            # p = 1 / (1 + alpha * mu); n = 1 / alpha
            nb_p = 1.0 / (1.0 + alpha_dispersion * mu)
            nb_n = 1.0 / alpha_dispersion
            sampled_cases = int(rng.negative_binomial(nb_n, nb_p))

            case_rows.append(
                {
                    "region_id": rid,
                    "month": m,
                    "cases": max(0, sampled_cases),
                    "definition": "clinical",
                    "synthetic": True,
                }
            )
            prev_y = sampled_cases

    cases_df = pd.DataFrame(case_rows)

    # Insert 1-2 intentional missing rows or values to check validate.py
    # Drop 2 random months in one region to test missing month detection
    target_region = region_ids[0]
    drop_idx = cases_df[
        (cases_df["region_id"] == target_region) & (cases_df["month"] == months[15])
    ].index
    if len(drop_idx) > 0:
        cases_df = cases_df.drop(drop_idx).reset_index(drop=True)

    result = {
        "cases": cases_df,
        "population": population_df,
        "climate_obs": climate_obs_df,
        "enso": enso_df,
        "landcover": landcover_df,
        "regions": regions_df,
    }

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        for tbl_name, df in result.items():
            write_table(df, tbl_name, output_dir=out_p, db_path=db_path)

        # Write ground truth metadata
        meta = get_synthetic_metadata()
        meta_file = out_p / "synthetic_metadata.json"
        meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return result


def get_synthetic_metadata() -> dict[str, Any]:
    """Return ground-truth coefficients and generator settings.

    Returns:
        Metadata dict for use in tests and model validation.
    """
    return {
        "coefficients": SYNTHETIC_COEFS,
        "n_regions": N_REGIONS,
        "n_years": N_YEARS,
        "start_year": START_YEAR,
        "description": (
            "Synthetic data for DengueCast testing. "
            "Ground-truth coefficients are stored here for test assertions."
        ),
    }
