"""Population data adapter with linear interpolation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from denguecast.data.schema import validate_population


def interpolate_population(
    df: pd.DataFrame, min_year: int = 2010, max_year: int = 2025
) -> pd.DataFrame:
    """Linearly interpolate and extrapolate annual population by region."""
    rows = []
    all_years = list(range(min_year, max_year + 1))

    for rid, group in df.groupby("region_id"):
        grp = group.sort_values("year").dropna(subset=["population"])
        if len(grp) >= 2:
            # Linear regression / interpolation
            interp_vals = np.interp(
                all_years,
                grp["year"].values,
                grp["population"].values,
            )
            for y, p in zip(all_years, interp_vals, strict=False):
                rows.append({"region_id": str(rid), "year": y, "population": int(round(p))})
        elif len(grp) == 1:
            base_pop = grp["population"].iloc[0]
            for y in all_years:
                rows.append({"region_id": str(rid), "year": y, "population": int(base_pop)})
        else:
            continue

    return pd.DataFrame(rows)


def load(path: str | Path, config: Any = None) -> pd.DataFrame:
    """Load and interpolate population data.

    Returns:
        DataFrame with columns [region_id, year, population].
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Population file not found: {p}")

    df = pd.read_csv(p)
    df["region_id"] = df["region_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["population"] = df["population"].astype(int)

    # If only sparse years provided (e.g. 2010, 2020), interpolate
    if len(df["year"].unique()) < 5:
        df = interpolate_population(df)

    validate_population(df)
    return df


def incidence(cases: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Compute incidence per 100,000 population.

    Returns:
        DataFrame with added 'incidence' and 'population' columns.
    """
    result = cases.copy()
    result["year"] = result["month"].astype(str).str.slice(0, 4).astype(int)

    merged = result.merge(
        pop[["region_id", "year", "population"]],
        on=["region_id", "year"],
        how="left",
    )

    merged["population"] = merged["population"].fillna(1_000_000).astype(int)
    merged["incidence"] = (100_000.0 * merged["cases"] / merged["population"]).round(3)
    return merged
