"""Hierarchical Bayesian model stub (Tahap B only).

Implements spatial-temporal NB model with:
- Region random effects
- Region x month AR1 effects
- Year effects
- Spatial ICAR/BYM2 effects

Same interface as NBGLM. Requires PyMC.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from denguecast.models.base import ForecastModel, ModelSpec


class HierarchicalBayes(ForecastModel):
    """STUB: Hierarchical Bayesian model for Tahap B."""

    def fit(self, train: pd.DataFrame) -> HierarchicalBayes:
        raise NotImplementedError("Tahap B: implement HierarchicalBayes with PyMC")

    def predict_samples(self, X: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> np.ndarray:
        raise NotImplementedError("Tahap B: implement HierarchicalBayes.predict_samples")

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(model_id="hierarchical_bayes", features=[])

    def to_dict(self) -> dict:
        raise NotImplementedError("Tahap B: implement HierarchicalBayes.to_dict")
