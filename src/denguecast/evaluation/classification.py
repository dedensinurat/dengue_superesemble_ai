"""Classification metrics for KLB outbreak exceedance (MODEL_SPEC C.11).

Evaluates binary probabilistic decisions against outbreak threshold exceedances:
- Contingency table: hits, false alarms, misses, correct rejections
- Sensitivity (Recall/TPR), PPV (Precision), Accuracy across cutoffs (0.25, 0.5, 0.75)
- Brier Score
- ROC AUC with bootstrap confidence intervals
- Data sample warnings when positive event count < 10
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def brier_score(p_prob: np.ndarray, y_binary: np.ndarray) -> float:
    """Compute Brier Score for probability forecasts of binary events.

    Brier = mean((p - y)^2). Lower is better (0 = perfect).
    """
    p = np.asarray(p_prob, dtype=float)
    y = np.asarray(y_binary, dtype=float)
    return float(np.mean((p - y) ** 2))


def contingency_table(
    p_prob: np.ndarray,
    y_binary: np.ndarray,
    cutoff: float = 0.5,
) -> dict[str, Any]:
    """Compute contingency table and performance statistics at a specific decision cutoff.

    Args:
        p_prob: Forecast probabilities in [0, 1].
        y_binary: True binary outcomes (0 or 1).
        cutoff: Decision threshold for predicting positive event (default: 0.5).

    Returns:
        Dictionary with hits, false_alarms, misses, correct_rejections,
        sensitivity, ppv, accuracy, brier, and data quality warnings.
    """
    p = np.asarray(p_prob, dtype=float)
    y = (np.asarray(y_binary) > 0).astype(int)
    pred = (p >= cutoff).astype(int)

    hits = int(np.sum((pred == 1) & (y == 1)))
    false_alarms = int(np.sum((pred == 1) & (y == 0)))
    misses = int(np.sum((pred == 0) & (y == 1)))
    correct_rejections = int(np.sum((pred == 0) & (y == 0)))
    total = len(y)

    n_positives = hits + misses
    n_predicted_positives = hits + false_alarms

    sensitivity = float(hits / n_positives) if n_positives > 0 else 0.0
    ppv = float(hits / n_predicted_positives) if n_predicted_positives > 0 else 0.0
    accuracy = float((hits + correct_rejections) / total) if total > 0 else 0.0
    brier = brier_score(p, y)

    warning = None
    if n_positives < 10:
        warning = f"Positive event count is low ({n_positives} < 10); metrics may have high variance."

    return {
        "cutoff": cutoff,
        "hits": hits,
        "false_alarms": false_alarms,
        "misses": misses,
        "correct_rejections": correct_rejections,
        "total": total,
        "n_positives": n_positives,
        "sensitivity": round(sensitivity, 4),
        "ppv": round(ppv, 4),
        "accuracy": round(accuracy, 4),
        "brier": round(brier, 4),
        "warning": warning,
    }


def compute_auc_with_ci(
    p_prob: np.ndarray,
    y_binary: np.ndarray,
    n_bootstraps: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Compute ROC AUC with bootstrap 95% confidence intervals.

    Returns:
        Dict with 'auc', 'ci_lower', 'ci_upper'.
    """
    p = np.asarray(p_prob, dtype=float)
    y = (np.asarray(y_binary) > 0).astype(int)

    if len(np.unique(y)) < 2:
        return {"auc": 0.5, "ci_lower": 0.5, "ci_upper": 0.5}

    try:
        point_auc = float(roc_auc_score(y, p))
    except Exception:
        point_auc = 0.5

    rng = np.random.default_rng(seed)
    n = len(y)
    boot_aucs: list[float] = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        y_boot = y[idx]
        if len(np.unique(y_boot)) < 2:
            continue
        with contextlib.suppress(Exception):
            boot_aucs.append(float(roc_auc_score(y_boot, p[idx])))

    if boot_aucs:
        ci_lower = float(np.percentile(boot_aucs, 2.5))
        ci_upper = float(np.percentile(boot_aucs, 97.5))
    else:
        ci_lower = point_auc
        ci_upper = point_auc

    return {
        "auc": round(point_auc, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def classification_table(
    p_exceed: np.ndarray,
    observed: np.ndarray,
    cutoff: float = 0.5,
) -> dict[str, Any]:
    """Compatibility interface for single-cutoff evaluation."""
    metrics = contingency_table(p_exceed, observed, cutoff=cutoff)
    auc_info = compute_auc_with_ci(p_exceed, observed)
    metrics.update(auc_info)
    return metrics
