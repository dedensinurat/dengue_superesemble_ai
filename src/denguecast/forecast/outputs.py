"""Forecast output formatting and persistence (MODEL_SPEC C.12 & C.13).

Formats probabilistic predictive draws into standardized tabular forecasts:
- Quantiles: mean, q025, q50, q975
- Exceedance probabilities: p_exceed_m1sd, p_exceed_m2sd, p_exceed_p75, p_exceed_p95
- Risk categorization: Rendah, Sedang, Tinggi, Sangat Tinggi
- Metadata: run_id, model_version, synthetic=True, reliable flag
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from denguecast.features.thresholds import exceedance_prob, risk_level

logger = logging.getLogger(__name__)


def format_forecasts(
    samples_by_region_lead: dict[tuple[str, int], np.ndarray],
    origin: str,
    channel_df: pd.DataFrame,
    pop_lookup: dict[tuple[str, int], int],
    run_id: str,
    config: Any = None,
    eligibility_lookup: dict[str, bool] | None = None,
    model_version: str = "0.1.0",
    synthetic: bool = True,
) -> pd.DataFrame:
    """Format probabilistic forecast samples into the canonical forecasts table.

    Args:
        samples_by_region_lead: Mapping of (region_id, lead) -> ndarray of shape (n_samples,).
        origin: Forecast origin month 'YYYY-MM'.
        channel_df: Endemic channel DataFrame per region and calendar month.
        pop_lookup: Mapping of (region_id, year) -> population count.
        run_id: Unique execution identifier.
        config: DengueCastConfig instance.
        eligibility_lookup: Optional mapping of region_id -> bool indicating reliability.
        model_version: Version string of forecasting system.
        synthetic: Boolean flag (strictly True for synthetic runs).

    Returns:
        DataFrame conforming to forecasts schema in C.12 and C.13.
    """
    origin_dt = pd.to_datetime(f"{origin}-01")
    records: list[dict[str, Any]] = []

    # Fast lookup for endemic channel
    ch_dict = {}
    if not channel_df.empty:
        for _, r in channel_df.iterrows():
            ch_dict[(str(r["region_id"]), int(r["month_of_year"]))] = r.to_dict()

    for (rid, lead), draws in samples_by_region_lead.items():
        draws = np.asarray(draws)
        target_dt = origin_dt + pd.DateOffset(months=lead)
        target_month = target_dt.strftime("%Y-%m")
        target_year = target_dt.year
        cal_m = target_dt.month

        # 1. Summary quantiles of counts
        mean_cnt = float(np.mean(draws))
        q025 = float(np.percentile(draws, 2.5))
        q50 = float(np.percentile(draws, 50.0))
        q975 = float(np.percentile(draws, 97.5))

        # 2. Convert to incidence per 100k
        pop = pop_lookup.get((rid, target_year), pop_lookup.get((rid, origin_dt.year), 1_000_000))
        inc_draws = 100_000.0 * draws / max(1, pop)
        inc_median = float(np.percentile(inc_draws, 50.0))

        # 3. Retrieve endemic channel thresholds
        ch = ch_dict.get((rid, cal_m), {})
        m1sd = ch.get("m1sd", np.percentile(inc_draws, 60.0))
        m2sd = ch.get("m2sd", np.percentile(inc_draws, 85.0))
        p75 = ch.get("p75", np.percentile(inc_draws, 75.0))
        p95 = ch.get("p95", np.percentile(inc_draws, 95.0))

        # 4. Calculate exceedance probabilities
        p_m1sd = exceedance_prob(inc_draws, m1sd)
        p_m2sd = exceedance_prob(inc_draws, m2sd)
        p_p75 = exceedance_prob(inc_draws, p75)
        p_p95 = exceedance_prob(inc_draws, p95)

        # 5. Categorize risk level
        r_level = risk_level(inc_median, ch)

        # 6. Check eligibility / reliability
        reliable = True
        if eligibility_lookup is not None:
            reliable = bool(eligibility_lookup.get(rid, True))

        records.append(
            {
                "run_id": run_id,
                "region_id": rid,
                "origin": origin,
                "target": target_month,
                "lead": lead,
                "mean": round(mean_cnt, 2),
                "q025": round(q025, 2),
                "q50": round(q50, 2),
                "q975": round(q975, 2),
                "p_exceed_m1sd": round(p_m1sd, 4),
                "p_exceed_m2sd": round(p_m2sd, 4),
                "p_exceed_p75": round(p_p75, 4),
                "p_exceed_p95": round(p_p95, 4),
                "risk_level": r_level,
                "model_version": model_version,
                "synthetic": synthetic,
                "reliable": reliable,
            }
        )

    df = pd.DataFrame(records)
    return df.sort_values(["region_id", "lead"]).reset_index(drop=True)
