"""OpenDengue CSV adapter for dengue case data (P3c).

Filters for Indonesia (34 provinces, pre-2022 boundaries).
Normalizes province names using data/ref/province_aliases.csv.
Aggregates weekly or daily records to calendar months.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from denguecast.data.schema import validate_cases

logger = logging.getLogger(__name__)


def load_province_aliases(path: str | Path = "data/ref/province_aliases.csv") -> dict[str, str]:
    """Load province alias lookup mapping."""
    p = Path(path)
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    return dict(zip(df["alias"].str.strip().str.lower(), df["region_id"].str.strip(), strict=False))


def normalize_province(name: str, alias_map: dict[str, str]) -> str | None:
    """Normalize a raw province name to a canonical region_id."""
    clean = str(name).strip().lower()
    return alias_map.get(clean)


def aggregate_weekly_to_monthly(df: pd.DataFrame, date_col: str, case_col: str) -> pd.DataFrame:
    """Aggregate weekly records to calendar months."""
    df = df.copy()
    dates = pd.to_datetime(df[date_col], errors="coerce")
    df["month"] = dates.dt.strftime("%Y-%m")
    df = df.dropna(subset=["month"])

    grouped = (
        df.groupby(["region_id", "month"])[case_col]
        .sum()
        .reset_index()
        .rename(columns={case_col: "cases"})
    )
    return grouped


def load(
    path: str | Path,
    config: Any = None,
    alias_path: str | Path = "data/ref/province_aliases.csv",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load OpenDengue CSV and filter for Indonesia province-level data.

    Returns:
        Tuple of (cases_df, quality_audit_dict).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"OpenDengue file not found: {p}")

    df = pd.read_csv(p, low_memory=False)

    # 1. Filter Indonesia if country column present
    for c_col in ["country", "country_code", "iso"]:
        if c_col in df.columns:
            df = df[df[c_col].astype(str).str.upper().isin(["IDN", "INDONESIA", "ID"])]
            break

    # 2. Identify province and date columns
    prov_col = None
    for col in ["adm_1_name", "admin1", "province", "provinsi", "region_name"]:
        if col in df.columns:
            prov_col = col
            break
    if prov_col is None:
        raise ValueError(f"Could not find province column in OpenDengue file: {list(df.columns)}")

    date_col = None
    for col in ["calendar_start_date", "date", "month", "start_date"]:
        if col in df.columns:
            date_col = col
            break
    if date_col is None:
        raise ValueError(f"Could not find date column in OpenDengue file: {list(df.columns)}")

    case_col = None
    for col in ["dengue_cases", "cases", "dengue_total", "count"]:
        if col in df.columns:
            case_col = col
            break
    if case_col is None:
        raise ValueError(f"Could not find cases column in OpenDengue file: {list(df.columns)}")

    # 3. Normalize province names
    alias_map = load_province_aliases(alias_path)
    df["region_id"] = df[prov_col].apply(lambda x: normalize_province(x, alias_map))

    unmapped = df[df["region_id"].isna()][prov_col].unique().tolist()
    if unmapped:
        logger.warning("Unmapped province names found in OpenDengue file: %s", unmapped)

    df = df.dropna(subset=["region_id"]).copy()

    # 4. Standardize date to month (YYYY-MM)
    dates = pd.to_datetime(df[date_col], errors="coerce")
    df["month"] = dates.dt.strftime("%Y-%m")
    df = df.dropna(subset=["month"]).copy()

    cases_df = (
        df.groupby(["region_id", "month"])[case_col]
        .sum()
        .reset_index()
        .rename(columns={case_col: "cases"})
    )

    # 5. Definition column
    def_col = None
    for col in ["definition", "type", "diagnosis"]:
        if col in df.columns:
            def_col = col
            break
    if def_col is not None and def_col in df.columns:
        defs = df.groupby(["region_id", "month"])[def_col].first().reset_index()
        cases_df = pd.merge(cases_df, defs, on=["region_id", "month"], how="left")
    else:
        cases_df["definition"] = "clinical"

    cases_df["cases"] = cases_df["cases"].fillna(0).astype(int)
    cases_df = cases_df.sort_values(["region_id", "month"]).reset_index(drop=True)

    # 6. Quality audit: suspicious consecutive zeros
    suspicious_provinces = []
    for rid, grp in cases_df.groupby("region_id"):
        consec_zeros = (
            (grp["cases"] == 0).astype(int).groupby((grp["cases"] != 0).cumsum()).cumsum().max()
        )
        if consec_zeros >= 6:
            suspicious_provinces.append((str(rid), int(consec_zeros)))

    audit = {
        "n_provinces": cases_df["region_id"].nunique(),
        "unmapped_provinces": unmapped,
        "suspicious_zero_runs": suspicious_provinces,
        "total_records": len(cases_df),
    }

    validate_cases(cases_df)
    return cases_df, audit
