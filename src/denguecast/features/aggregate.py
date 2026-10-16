"""Population-weighted aggregation of climate grid/point data to regions."""

from __future__ import annotations

import numpy as np
import pandas as pd


def aggregate_to_regions(
    cells: pd.DataFrame,
    weights: pd.DataFrame,
    val_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Aggregate cell/point-level data to regions using population weights.

    Args:
        cells: DataFrame with [region_id, cell_id, ...value_cols].
        weights: DataFrame with [region_id, cell_id, weight].
        val_cols: Columns to aggregate (default: all numeric columns in cells).

    Returns:
        Region-level DataFrame with weighted average values.
    """
    merged = pd.merge(cells, weights, on=["region_id", "cell_id"])
    if merged.empty:
        raise ValueError("No common (region_id, cell_id) pairs found between cells and weights.")

    if val_cols is None:
        val_cols = [
            c
            for c in cells.select_dtypes(include=[np.number]).columns
            if c not in ["region_id", "cell_id", "weight"]
        ]

    rows = []
    for rid, grp in merged.groupby("region_id"):
        w = grp["weight"].values.astype(float)
        w_sum = w.sum()
        if w_sum <= 0:
            # Fallback to unweighted mean
            w = np.ones_like(w)
            w_sum = w.sum()

        row: dict[str, object] = {"region_id": rid}
        for col in val_cols:
            vals = grp[col].values.astype(float)
            valid = ~np.isnan(vals)
            if valid.any():
                w_valid = w[valid]
                w_v_sum = w_valid.sum()
                if w_v_sum > 0:
                    row[col] = float(np.sum(vals[valid] * w_valid) / w_v_sum)
                else:
                    row[col] = float(np.mean(vals[valid]))
            else:
                row[col] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def compute_dtr(df: pd.DataFrame) -> pd.DataFrame:
    """Compute diurnal temperature range: dtr = tmax - tmin.

    Args:
        df: DataFrame with tmax and tmin columns.

    Returns:
        DataFrame with added or updated 'dtr' column.
    """
    res = df.copy()
    if "tmax" in res.columns and "tmin" in res.columns:
        res["dtr"] = (res["tmax"] - res["tmin"]).round(2)
        res.loc[res["dtr"] < 0, "dtr"] = np.nan
    return res
