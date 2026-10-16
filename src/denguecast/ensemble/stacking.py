"""Superensemble: CRPS-weighted linear pool (MODEL_SPEC C.8).

Combines probabilistic forecasts from candidate models and baselines:
- Weights: w_i = (1 / CRPS_i^2) / sum_j(1 / CRPS_j^2)
- Linear pool: draws round(w_i * N) samples from model i, summing exactly to N.
- Residual rounding error is added to the model with the largest weight.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from denguecast.models.base import ForecastModel, ModelSpec


def compute_weights(
    crps_by_model: pd.DataFrame | pd.Series | dict[str, float],
) -> pd.Series:
    """Compute CRPS-squared inverse weights.

    w_i = (1 / CRPS_i^2) / sum_j (1 / CRPS_j^2)

    Args:
        crps_by_model: Mapping or Series/DataFrame with model_id and mean CRPS.

    Returns:
        Series of weights indexed by model_id, summing to 1.0, sorted descending.
    """
    if isinstance(crps_by_model, pd.DataFrame):
        if "model_id" in crps_by_model.columns and "crps" in crps_by_model.columns:
            series = crps_by_model.groupby("model_id")["crps"].mean()
        elif "model_id" in crps_by_model.columns and "crps_mean" in crps_by_model.columns:
            series = crps_by_model.groupby("model_id")["crps_mean"].mean()
        else:
            series = crps_by_model.iloc[:, 0]
    elif isinstance(crps_by_model, dict):
        series = pd.Series(crps_by_model)
    elif isinstance(crps_by_model, pd.Series):
        series = crps_by_model.copy()
    else:
        raise TypeError(f"Unsupported input type for crps_by_model: {type(crps_by_model)}")

    # Ensure clean non-zero values
    crps_vals = np.maximum(1e-6, series.astype(float).values)
    inv_sq = 1.0 / (crps_vals**2)
    weights = inv_sq / np.sum(inv_sq)

    result_series = pd.Series(weights, index=series.index, name="weight").sort_values(
        ascending=False
    )
    return result_series


class Superensemble(ForecastModel):
    """Linear pool of K models weighted inversely by CRPS^2."""

    def __init__(
        self,
        models: list[ForecastModel],
        weights: pd.Series | dict[str, float],
    ) -> None:
        self.models = models
        if isinstance(weights, dict):
            self.weights = pd.Series(weights)
        else:
            self.weights = weights.copy()

        # Align weights with models
        model_ids = [m.spec.model_id for m in models]
        clean_weights = np.array([float(self.weights.get(mid, 1.0 / len(models))) for mid in model_ids])
        if np.sum(clean_weights) > 0:
            clean_weights = clean_weights / np.sum(clean_weights)
        else:
            clean_weights = np.ones(len(models)) / len(models)
        self.weights = pd.Series(clean_weights, index=model_ids)

    def fit(self, train: pd.DataFrame) -> Superensemble:
        """Fit all constituent models if not already fit."""
        for m in self.models:
            m.fit(train)
        return self

    def predict_samples(
        self,
        X: pd.DataFrame,
        n_samples: int = 1000,
        seed: int = 42,
        combine: str = "linear_pool",
    ) -> np.ndarray:
        """Sample from linear pool proportional to weights.

        Args:
            X: Feature DataFrame.
            n_samples: Total number of predictive draws (default 1000).
            seed: Random seed.
            combine: 'linear_pool' (default) or 'weighted_mean_samples'.

        Returns:
            Array of shape [n_rows, n_samples] with non-negative integer counts.
        """
        n_rows = len(X)
        if not self.models:
            return np.zeros((n_rows, n_samples), dtype=int)

        w_vals = self.weights.values

        if combine == "weighted_mean_samples":
            accum = np.zeros((n_rows, n_samples), dtype=float)
            for idx, m in enumerate(self.models):
                s = m.predict_samples(X, n_samples=n_samples, seed=seed + idx)
                accum += w_vals[idx] * s
            return np.maximum(0, np.round(accum)).astype(int)

        # Standard Linear Pool: allocate sample counts per model
        # 1. Compute proportional counts
        raw_allocations = np.round(w_vals * n_samples).astype(int)

        # 2. Adjust discrepancy to match exactly n_samples
        diff = n_samples - int(np.sum(raw_allocations))
        if diff != 0:
            # Add/subtract remainder from model with highest weight
            max_w_idx = int(np.argmax(w_vals))
            raw_allocations[max_w_idx] += diff

        # 3. Draw samples from each model and concatenate
        collected_samples: list[np.ndarray] = []
        for idx, m in enumerate(self.models):
            count_i = raw_allocations[idx]
            if count_i > 0:
                s_i = m.predict_samples(X, n_samples=count_i, seed=seed + idx * 17)
                collected_samples.append(s_i)

        if not collected_samples:
            return np.zeros((n_rows, n_samples), dtype=int)

        pooled = np.concatenate(collected_samples, axis=1)

        # 4. Deterministic shuffle of columns for clean uniform mixture
        rng = np.random.default_rng(seed)
        shuffled_cols = rng.permutation(pooled.shape[1])
        return pooled[:, shuffled_cols]

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(
            model_id="superensemble",
            features=[],
            description=f"Superensemble linear pool of {len(self.models)} models",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": "superensemble",
            "type": "Superensemble",
            "n_models": len(self.models),
            "weights": self.weights.to_dict(),
        }
