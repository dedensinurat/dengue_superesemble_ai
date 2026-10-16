"""Schema validation for all DengueCast data tables.

Validates structure, types, and value ranges for:
- cases(region_id, month, cases, definition)
- population(region_id, year, population)
- climate_obs(region_id, month, tmean, tmin, tmax, dtr, precip, humidity, wind)
- enso(month, nino34_anom)
- landcover(region_id, year, urban_pct, periurban_pct)

Indonesia extensions (C.13):
- climate_obs also has: source, n_stations, valid_frac, quant_error
- regions(region_id, name, region_group, lat, lon)
- forecasts.reliable
- province_eligibility(run_id, region_id, crpss_median, ci_low, ci_high, eligible)
"""

from __future__ import annotations

import re

import pandas as pd

# Temperature bounds (°C)
TEMP_MIN: float = -10.0
TEMP_MAX: float = 50.0

# Humidity bounds (%)
HUMIDITY_MIN: float = 0.0
HUMIDITY_MAX: float = 100.0

_MONTH_REGEX = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")


def _check_required_columns(df: pd.DataFrame, required: list[str], table_name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Table '{table_name}' missing required columns: {missing}. Present: {list(df.columns)}"
        )


def _check_month_format(series: pd.Series, table_name: str) -> None:
    non_null = series.dropna().astype(str)
    invalid = non_null[~non_null.str.match(_MONTH_REGEX)]
    if not invalid.empty:
        raise ValueError(
            f"Table '{table_name}' contains invalid month format (expected YYYY-MM): {invalid.head(5).tolist()}"
        )


def validate_cases(df: pd.DataFrame) -> None:
    """Validate cases table schema and value ranges.

    Args:
        df: DataFrame with columns [region_id, month, cases, definition].

    Raises:
        ValueError: If schema or values are invalid.
    """
    _check_required_columns(df, ["region_id", "month", "cases", "definition"], "cases")
    _check_month_format(df["month"], "cases")

    # Check duplicates on (region_id, month)
    dups = df.duplicated(subset=["region_id", "month"])
    if dups.any():
        dup_rows = df[dups][["region_id", "month"]].head(5).to_dict("records")
        raise ValueError(f"Table 'cases' contains duplicate (region_id, month) entries: {dup_rows}")

    # Check non-negative cases
    non_null_cases = df["cases"].dropna()
    if (non_null_cases < 0).any():
        invalid = non_null_cases[non_null_cases < 0].head(5).tolist()
        raise ValueError(f"Table 'cases' contains negative case counts: {invalid}")


def validate_population(df: pd.DataFrame) -> None:
    """Validate population table schema.

    Args:
        df: DataFrame with columns [region_id, year, population].

    Raises:
        ValueError: If schema or values are invalid.
    """
    _check_required_columns(df, ["region_id", "year", "population"], "population")

    dups = df.duplicated(subset=["region_id", "year"])
    if dups.any():
        dup_rows = df[dups][["region_id", "year"]].head(5).to_dict("records")
        raise ValueError(
            f"Table 'population' contains duplicate (region_id, year) entries: {dup_rows}"
        )

    non_null_pop = df["population"].dropna()
    if (non_null_pop <= 0).any():
        invalid = non_null_pop[non_null_pop <= 0].head(5).tolist()
        raise ValueError(f"Table 'population' contains non-positive population: {invalid}")


def validate_climate_obs(df: pd.DataFrame) -> None:
    """Validate climate_obs table schema and value ranges.

    Checks temperature bounds [-10, 50], humidity [0, 100], precip >= 0.
    Canonical column names only (tmin, tmax, tmean, humidity, precip).
    Source column names (Tn, Tx, etc.) must NOT appear here.

    Args:
        df: DataFrame with canonical climate columns.

    Raises:
        ValueError: If schema or values are invalid.
    """
    # Prohibit non-canonical source column names
    forbidden_cols = [
        "Tn",
        "Tx",
        "Tavg",
        "RH_avg",
        "RR",
        "ss",
        "ff_x",
        "ddd_x",
        "ff_avg",
        "ddd_car",
    ]
    present_forbidden = [c for c in forbidden_cols if c in df.columns]
    if present_forbidden:
        raise ValueError(
            f"Table 'climate_obs' contains source column names: {present_forbidden}. Canonical names expected!"
        )

    required = ["region_id", "month", "tmean", "tmin", "tmax", "dtr", "precip", "humidity", "wind"]
    _check_required_columns(df, required, "climate_obs")
    _check_month_format(df["month"], "climate_obs")

    dups = df.duplicated(subset=["region_id", "month"])
    if dups.any():
        dup_rows = df[dups][["region_id", "month"]].head(5).to_dict("records")
        raise ValueError(
            f"Table 'climate_obs' contains duplicate (region_id, month) entries: {dup_rows}"
        )

    # Temperature bounds
    for col in ["tmean", "tmin", "tmax"]:
        vals = df[col].dropna()
        if (vals < TEMP_MIN).any() or (vals > TEMP_MAX).any():
            invalid = vals[(vals < TEMP_MIN) | (vals > TEMP_MAX)].head(5).tolist()
            raise ValueError(
                f"Column '{col}' in 'climate_obs' outside bounds [{TEMP_MIN}, {TEMP_MAX}]: {invalid}"
            )

    # Humidity bounds [0, 100]
    hum = df["humidity"].dropna()
    if (hum < HUMIDITY_MIN).any() or (hum > HUMIDITY_MAX).any():
        invalid = hum[(hum < HUMIDITY_MIN) | (hum > HUMIDITY_MAX)].head(5).tolist()
        raise ValueError(
            f"Column 'humidity' in 'climate_obs' outside bounds [{HUMIDITY_MIN}, {HUMIDITY_MAX}]: {invalid}"
        )

    # Precip >= 0
    precip = df["precip"].dropna()
    if (precip < 0).any():
        invalid = precip[precip < 0].head(5).tolist()
        raise ValueError(f"Column 'precip' in 'climate_obs' contains negative values: {invalid}")

    # Wind >= 0
    wind = df["wind"].dropna()
    if (wind < 0).any():
        invalid = wind[wind < 0].head(5).tolist()
        raise ValueError(f"Column 'wind' in 'climate_obs' contains negative values: {invalid}")

    # DTR >= 0 (allow NaN)
    dtr = df["dtr"].dropna()
    if (dtr < 0).any():
        invalid = dtr[dtr < 0].head(5).tolist()
        raise ValueError(f"Column 'dtr' in 'climate_obs' contains negative values: {invalid}")


def validate_enso(df: pd.DataFrame) -> None:
    """Validate enso table schema.

    Args:
        df: DataFrame with columns [month, nino34_anom].

    Raises:
        ValueError: If schema or values are invalid.
    """
    _check_required_columns(df, ["month", "nino34_anom"], "enso")
    _check_month_format(df["month"], "enso")

    dups = df.duplicated(subset=["month"])
    if dups.any():
        dup_rows = df[dups]["month"].head(5).tolist()
        raise ValueError(f"Table 'enso' contains duplicate months: {dup_rows}")


def validate_landcover(df: pd.DataFrame) -> None:
    """Validate landcover table schema.

    Args:
        df: DataFrame with columns [region_id, year, urban_pct, periurban_pct].

    Raises:
        ValueError: If schema or values are invalid.
    """
    _check_required_columns(df, ["region_id", "year", "urban_pct", "periurban_pct"], "landcover")

    dups = df.duplicated(subset=["region_id", "year"])
    if dups.any():
        dup_rows = df[dups][["region_id", "year"]].head(5).to_dict("records")
        raise ValueError(
            f"Table 'landcover' contains duplicate (region_id, year) entries: {dup_rows}"
        )

    for col in ["urban_pct", "periurban_pct"]:
        vals = df[col].dropna()
        if (vals < 0).any() or (vals > 100).any():
            invalid = vals[(vals < 0) | (vals > 100)].head(5).tolist()
            raise ValueError(f"Column '{col}' in 'landcover' outside bounds [0, 100]: {invalid}")
