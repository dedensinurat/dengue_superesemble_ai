"""Probabilistic scoring metrics (MODEL_SPEC C.11).

Implements:
- CRPS (Continuous Ranked Probability Score) from samples via energy score estimator.
- CRPSS (CRPS Skill Score relative to baseline).
- Standard classification and point metrics (MAE, RMSE, Bias, Coverage, Interval Width).
"""

from __future__ import annotations

import numpy as np


def crps_samples(samples: np.ndarray, y: float | np.ndarray) -> float | np.ndarray:
    """Compute CRPS from predictive samples using the energy score estimator.

    Formula:
        CRPS = E|X - y| - 0.5 * E|X - X'|
        For sorted samples x_(1) <= ... <= x_(m):
        E|X - y| = mean(|X - y|)
        0.5 * E|X - X'| = sum_i ((2i - m - 1) * x_(i)) / m^2

    Args:
        samples: Array of forecast samples. Shape (n_samples,) or (n_rows, n_samples).
        y: Observed value(s). Scalar or array of shape (n_rows,).

    Returns:
        CRPS value(s), lower is better. Float if 1D input, 1D array if 2D input.
    """
    samples = np.asarray(samples, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if samples.ndim == 1:
        m = len(samples)
        if m == 0:
            return 0.0
        sorted_x = np.sort(samples)
        t1 = np.mean(np.abs(samples - float(y_arr)))
        weights = 2 * np.arange(1, m + 1) - m - 1
        t2 = np.sum(sorted_x * weights) / (m * m)
        return float(max(0.0, t1 - t2))

    elif samples.ndim == 2:
        _n, m = samples.shape
        if m == 0:
            return np.zeros(len(y_arr))
        sorted_x = np.sort(samples, axis=1)
        y_vec = y_arr if y_arr.ndim == 1 else y_arr.squeeze()
        t1 = np.mean(np.abs(samples - y_vec[:, None]), axis=1)
        weights = 2 * np.arange(1, m + 1) - m - 1
        t2 = np.sum(sorted_x * weights[None, :], axis=1) / (m * m)
        return np.maximum(0.0, t1 - t2)

    else:
        raise ValueError(f"Expected 1D or 2D samples, got shape {samples.shape}")


def crpss(crps_model: float, crps_baseline: float) -> float:
    """Compute CRPS Skill Score relative to baseline.

    CRPSS = 1 - (CRPS_model / CRPS_baseline)
    - 1.0 = perfect forecast
    - > 0.0 = outperforms baseline
    - 0.0 = same skill as baseline
    - < 0.0 = worse than baseline
    """
    if crps_baseline <= 1e-9:
        return 0.0
    return float(1.0 - (crps_model / crps_baseline))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Forecast Bias (mean error = pred - true)."""
    return float(np.mean(np.asarray(y_pred) - np.asarray(y_true)))


def coverage_95(samples: np.ndarray, y: float | np.ndarray) -> float:
    """Compute empirical coverage of the 95% prediction interval [q025, q975]."""
    samples = np.asarray(samples)
    y_arr = np.asarray(y)

    if samples.ndim == 1:
        q025 = np.percentile(samples, 2.5)
        q975 = np.percentile(samples, 97.5)
        return 1.0 if q025 <= float(y_arr) <= q975 else 0.0
    elif samples.ndim == 2:
        q025 = np.percentile(samples, 2.5, axis=1)
        q975 = np.percentile(samples, 97.5, axis=1)
        y_vec = y_arr if y_arr.ndim == 1 else y_arr.squeeze()
        covered = (y_vec >= q025) & (y_vec <= q975)
        return float(np.mean(covered))
    else:
        raise ValueError(f"Expected 1D or 2D samples, got {samples.shape}")


def interval_width_95(samples: np.ndarray) -> float:
    """Compute average width of the 95% prediction interval (q975 - q025)."""
    samples = np.asarray(samples)
    if samples.ndim == 1:
        return float(np.percentile(samples, 97.5) - np.percentile(samples, 2.5))
    elif samples.ndim == 2:
        widths = np.percentile(samples, 97.5, axis=1) - np.percentile(samples, 2.5, axis=1)
        return float(np.mean(widths))
    else:
        raise ValueError(f"Expected 1D or 2D samples, got {samples.shape}")
