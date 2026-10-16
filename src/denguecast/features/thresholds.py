"""Endemic channel computation and risk classification (MODEL_SPEC C.10)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def endemic_channel(
    inc_df: pd.DataFrame,
    origin: str,
    years: int = 5,
    exclude_outbreak_years: bool = False,
) -> pd.DataFrame:
    """Compute endemic channel from historical incidence per region and calendar month.

    STRICT ANTI-LEAKAGE: Uses ONLY data with month <= origin.

    Args:
        inc_df: DataFrame containing [region_id, month, incidence].
        origin: Forecast origin string (YYYY-MM).
        years: Number of historical years before origin to use (default 5).
        exclude_outbreak_years: If True, exclude years exceeding mean+2SD annual incidence.

    Returns:
        DataFrame [region_id, month_of_year, mean, sd, m1sd, m2sd, p75, p95].
    """
    hist = inc_df[inc_df["month"] <= origin].copy()
    if hist.empty:
        raise ValueError(f"No historical incidence available on or before origin '{origin}'")

    hist["year"] = hist["month"].str.slice(0, 4).astype(int)
    hist["month_of_year"] = hist["month"].str.slice(5, 7).astype(int)

    origin_year = int(origin.split("-")[0])
    min_year = origin_year - years
    hist = hist[hist["year"] >= min_year]

    if exclude_outbreak_years and len(hist["year"].unique()) > 3:
        # Identify outbreak years per region
        ann_inc = hist.groupby(["region_id", "year"])["incidence"].sum().reset_index()
        keep_years = set()
        for _rid, grp in ann_inc.groupby("region_id"):
            m = grp["incidence"].mean()
            s = grp["incidence"].std()
            normal_years = grp[grp["incidence"] <= m + 2.0 * s]["year"].tolist()
            keep_years.update(normal_years)
        hist = hist[hist["year"].isin(keep_years)]

    rows = []
    for (rid, moy), grp in hist.groupby(["region_id", "month_of_year"]):
        vals = grp["incidence"].dropna().values
        if len(vals) == 0:
            mean_val, sd_val, p75_val, p95_val = 0.0, 1.0, 0.0, 0.0
        else:
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else max(1.0, mean_val * 0.3)
            p75_val = float(np.percentile(vals, 75))
            p95_val = float(np.percentile(vals, 95))

        rows.append(
            {
                "region_id": str(rid),
                "month_of_year": int(moy),
                "mean": round(mean_val, 3),
                "sd": round(sd_val, 3),
                "m1sd": round(mean_val + sd_val, 3),
                "m2sd": round(mean_val + 2.0 * sd_val, 3),
                "p75": round(p75_val, 3),
                "p95": round(p95_val, 3),
            }
        )

    return pd.DataFrame(rows)


def risk_level(inc: float, channel: pd.Series | dict[str, Any]) -> str:
    """Classify incidence into risk level according to endemic channel thresholds.

    Threshold rules:
    - rendah:       < mean
    - sedang:       mean <= inc < mean + 1*sd
    - tinggi:       mean + 1*sd <= inc < mean + 2*sd
    - sangat_tinggi: >= mean + 2*sd

    Returns:
        One of: 'rendah' | 'sedang' | 'tinggi' | 'sangat_tinggi'.
    """
    mean_val = float(channel["mean"])
    m1sd = float(channel.get("m1sd", mean_val + float(channel.get("sd", 0.0))))
    m2sd = float(channel.get("m2sd", mean_val + 2.0 * float(channel.get("sd", 0.0))))

    if inc < mean_val:
        return "rendah"
    if inc < m1sd:
        return "sedang"
    if inc < m2sd:
        return "tinggi"
    return "sangat_tinggi"


def exceedance_prob(samples_inc: np.ndarray, threshold: float) -> float:
    """Compute exceedance probability from samples.

    Returns:
        Fraction of samples exceeding threshold [0.0, 1.0].
    """
    arr = np.asarray(samples_inc, dtype=float)
    if len(arr) == 0:
        return 0.0
    return float(np.mean(arr > threshold))
