"""Abstract ForecastModel interface (MODEL_SPEC C.3)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ModelSpec:
    """Specification for a candidate model."""

    model_id: str
    features: list[str] = field(default_factory=list)
    climate_by_group: bool = False
    description: str = ""


class ForecastModel(ABC):
    """Abstract base class for all DengueCast forecast models."""

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> ForecastModel:
        """Fit model on training data.

        Args:
            train: Training feature DataFrame.

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict_samples(self, X: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> np.ndarray:
        """Generate probabilistic forecast samples.

        Args:
            X: Feature DataFrame (one row per region).
            n_samples: Number of samples to draw.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape [n_rows, n_samples] with non-negative integer counts.
        """
        ...

    @property
    @abstractmethod
    def spec(self) -> ModelSpec:
        """Return model specification."""
        ...

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize model metadata to dict."""
        ...
