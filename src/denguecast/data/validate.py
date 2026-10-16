"""Data quality validation and reporting for DengueCast.

Produces a quality report including:
- Missing months per region
- Duplicate records
- Outliers (|z| > 5)
- Regions with fewer than N months of data
- Possibly incomplete last month (reporting delay)
"""

from __future__ import annotations

import pandas as pd


def _generate_expected_months(start_month: str, end_month: str) -> list[str]:
    """Generate list of 'YYYY-MM' strings between start and end inclusive."""
    dates = pd.date_range(start=f"{start_month}-01", end=f"{end_month}-01", freq="MS")
    return dates.strftime("%Y-%m").tolist()


def quality_report(
    cases: pd.DataFrame,
    climate_obs: pd.DataFrame,
    min_months: int = 24,
) -> tuple[pd.DataFrame, str]:
    """Generate a data quality report.

    Args:
        cases: Cases table [region_id, month, cases, definition].
        climate_obs: Climate observations table [region_id, month, tmean, ...].
        min_months: Minimum months required per region.

    Returns:
        Tuple of (issues_df, summary_text).
    """
    issues: list[dict[str, str]] = []

    # 1. Check cases duplicates
    if "region_id" in cases.columns and "month" in cases.columns:
        case_dups = cases.duplicated(subset=["region_id", "month"], keep=False)
        if case_dups.any():
            dup_count = case_dups.sum()
            issues.append(
                {
                    "table": "cases",
                    "region_id": "ALL",
                    "issue_type": "duplicate_records",
                    "description": f"Found {dup_count} duplicate (region_id, month) rows in cases",
                    "severity": "ERROR",
                }
            )

        # Missing months and record counts per region
        for rid, group in cases.groupby("region_id"):
            n_records = len(group)
            if n_records < min_months:
                issues.append(
                    {
                        "table": "cases",
                        "region_id": str(rid),
                        "issue_type": "insufficient_data",
                        "description": f"Only {n_records} months of data (required >= {min_months})",
                        "severity": "WARNING",
                    }
                )

            months = sorted(group["month"].dropna().unique())
            if len(months) >= 2:
                expected = _generate_expected_months(months[0], months[-1])
                missing = set(expected) - set(months)
                if missing:
                    issues.append(
                        {
                            "table": "cases",
                            "region_id": str(rid),
                            "issue_type": "missing_months",
                            "description": f"Missing {len(missing)} months between {months[0]} and {months[-1]}: {sorted(missing)[:5]}...",
                            "severity": "WARNING",
                        }
                    )

        # Outliers in cases (|z| > 5)
        if "cases" in cases.columns:
            case_vals = cases["cases"].dropna()
            if len(case_vals) > 10 and case_vals.std() > 0:
                z = (case_vals - case_vals.mean()) / case_vals.std()
                outliers = cases.loc[z.abs() > 5]
                for _, row in outliers.iterrows():
                    issues.append(
                        {
                            "table": "cases",
                            "region_id": str(row["region_id"]),
                            "issue_type": "outlier",
                            "description": f"Outlier cases count: {row['cases']} at month {row['month']}",
                            "severity": "WARNING",
                        }
                    )

    # 2. Check climate duplicates and missing values
    if "region_id" in climate_obs.columns and "month" in climate_obs.columns:
        clim_dups = climate_obs.duplicated(subset=["region_id", "month"], keep=False)
        if clim_dups.any():
            issues.append(
                {
                    "table": "climate_obs",
                    "region_id": "ALL",
                    "issue_type": "duplicate_records",
                    "description": f"Found {clim_dups.sum()} duplicate rows in climate_obs",
                    "severity": "ERROR",
                }
            )

        # Check missing values in climate variables
        clim_vars = [
            c
            for c in ["tmean", "tmin", "tmax", "precip", "humidity", "wind"]
            if c in climate_obs.columns
        ]
        for col in clim_vars:
            n_nan = climate_obs[col].isna().sum()
            if n_nan > 0:
                issues.append(
                    {
                        "table": "climate_obs",
                        "region_id": "ALL",
                        "issue_type": "missing_values",
                        "description": f"Variable '{col}' has {n_nan} missing values ({n_nan/len(climate_obs)*100:.1f}%)",
                        "severity": "INFO",
                    }
                )

            # Check outliers (|z| > 5)
            vals = climate_obs[col].dropna()
            if len(vals) > 10 and vals.std() > 0:
                z = (vals - vals.mean()) / vals.std()
                n_out = (z.abs() > 5).sum()
                if n_out > 0:
                    issues.append(
                        {
                            "table": "climate_obs",
                            "region_id": "ALL",
                            "issue_type": "outlier",
                            "description": f"Variable '{col}' has {n_out} extreme outliers (|z| > 5)",
                            "severity": "WARNING",
                        }
                    )

    issues_df = pd.DataFrame(
        issues,
        columns=["table", "region_id", "issue_type", "description", "severity"],
    )

    # Generate summary text
    lines = [
        "=== DengueCast Data Quality Report ===",
        f"Total issues identified: {len(issues_df)}",
        f"Errors: {(issues_df['severity'] == 'ERROR').sum()}",
        f"Warnings: {(issues_df['severity'] == 'WARNING').sum()}",
        f"Info: {(issues_df['severity'] == 'INFO').sum()}",
    ]
    if not issues_df.empty:
        lines.append("\nBreakdown by issue type:")
        for itype, count in issues_df["issue_type"].value_counts().items():
            lines.append(f"  - {itype}: {count}")

    summary_text = "\n".join(lines)
    return issues_df, summary_text


def flag_possibly_incomplete(
    df: pd.DataFrame,
    month_col: str = "month",
) -> pd.DataFrame:
    """Flag the latest month as possibly_incomplete due to reporting delay.

    Args:
        df: DataFrame with a month column.
        month_col: Name of the month column.

    Returns:
        DataFrame with added 'possibly_incomplete' boolean column.
    """
    result = df.copy()
    if month_col not in result.columns or result.empty:
        result["possibly_incomplete"] = False
        return result

    latest_month = result[month_col].dropna().max()
    result["possibly_incomplete"] = result[month_col] == latest_month
    return result
