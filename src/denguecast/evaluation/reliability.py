"""Reliability diagram data for probabilistic outbreak forecasts (MODEL_SPEC C.11).

Binned evaluation of forecast probabilities vs observed event frequencies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def reliability_table(
    p_exceed: np.ndarray,
    observed: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute reliability table for binned calibration assessment.

    Args:
        p_exceed: Predicted exceedance probabilities in [0, 1].
        observed: Observed binary outbreak events (0 or 1).
        n_bins: Number of probability bins (default: 10).

    Returns:
        DataFrame with columns [bin_lower, bin_upper, bin_center, mean_forecast_prob, observed_freq, count].
    """
    p = np.asarray(p_exceed, dtype=float)
    y = (np.asarray(observed) > 0).astype(int)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict[str, float]] = []

    for i in range(n_bins):
        low = bin_edges[i]
        high = bin_edges[i + 1]

        mask = (p >= low) & (p <= high) if i == n_bins - 1 else (p >= low) & (p < high)

        count = int(np.sum(mask))
        center = (low + high) / 2.0

        if count > 0:
            mean_prob = float(np.mean(p[mask]))
            obs_freq = float(np.mean(y[mask]))
        else:
            mean_prob = center
            obs_freq = np.nan

        rows.append(
            {
                "bin_lower": round(float(low), 3),
                "bin_upper": round(float(high), 3),
                "bin_center": round(float(center), 3),
                "mean_forecast_prob": round(mean_prob, 4),
                "observed_freq": round(obs_freq, 4) if not np.isnan(obs_freq) else np.nan,
                "count": count,
            }
        )

    return pd.DataFrame(rows)
