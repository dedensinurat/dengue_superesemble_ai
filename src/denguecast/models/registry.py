"""Model registry: creates ForecastModel instances from ModelSpec."""

from __future__ import annotations

from denguecast.models.base import ForecastModel, ModelSpec
from denguecast.models.baseline import SeasonalExpandingMean, SeasonalNB
from denguecast.models.hierarchical_bayes import HierarchicalBayes
from denguecast.models.nb_glm import NBGLM


def build_model(spec: ModelSpec) -> ForecastModel:
    """Instantiate an unfitted model from its ModelSpec.

    Args:
        spec: ModelSpec with model_id and features.

    Returns:
        Unfitted ForecastModel instance.
    """
    mid = spec.model_id.lower()
    if mid in ["b1", "b1_seasonal_mean", "seasonal_mean"]:
        return SeasonalExpandingMean()
    elif mid in ["b2", "b2_seasonal_nb", "seasonal_nb"]:
        return SeasonalNB()
    elif mid.startswith("nbglm") or mid.startswith("nb_glm"):
        return NBGLM(
            features=spec.features,
            climate_by_group=spec.climate_by_group,
            model_id=spec.model_id,
        )
    elif mid in ["hierarchical_bayes", "bayes"]:
        return HierarchicalBayes()
    else:
        # Fallback to NBGLM with given features
        return NBGLM(
            features=spec.features,
            climate_by_group=spec.climate_by_group,
            model_id=spec.model_id,
        )


def list_models() -> list[str]:
    """List available model types."""
    return ["B1_seasonal_mean", "B2_seasonal_nb", "nbglm", "hierarchical_bayes"]
