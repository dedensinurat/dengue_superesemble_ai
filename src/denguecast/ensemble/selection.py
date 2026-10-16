"""Candidate scoring and selection via expanding-window CRPS (MODEL_SPEC C.7).

Evaluates candidate models across expanding-window walk-forward splits,
scores them by CRPS, and selects the top K models plus baselines.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from denguecast.evaluation.scores import crps_samples
from denguecast.evaluation.splits import ExpandingWindowSplitter
from denguecast.features.build import build_training_frame
from denguecast.models.base import ModelSpec
from denguecast.models.registry import build_model

logger = logging.getLogger(__name__)

# In-memory score cache by hash
_SCORE_CACHE: dict[str, float] = {}


def _score_single_spec(
    spec: ModelSpec,
    train_df: pd.DataFrame,
    splits: list[Any],
    n_samples: int = 150,
) -> list[dict[str, Any]]:
    """Score one specification across expanding window splits."""
    records: list[dict[str, Any]] = []

    for sp in splits:
        # Strict anti-leakage: training data only up to origin
        train_split = train_df[train_df["month"].isin(sp.train_months)].copy()
        if train_split.empty or train_split["region_id"].nunique() < 2:
            continue

        try:
            model = build_model(spec)
            model.fit(train_split)
        except Exception as e:
            logger.debug(f"Failed to fit spec {spec.model_id} on split {sp.origin}: {e}")
            continue

        for target_month, lead in sp.test_targets:
            test_sub = train_df[train_df["month"] == target_month].copy()
            if test_sub.empty:
                continue

            # Cache key
            feat_key = f"{spec.model_id}_{target_month}_{lead}_{len(train_split)}"
            if feat_key in _SCORE_CACHE:
                crps_val = _SCORE_CACHE[feat_key]
            else:
                try:
                    samples = model.predict_samples(test_sub, n_samples=n_samples, seed=42 + lead)
                    y_obs = test_sub["cases"].values
                    crps_val = float(np.mean(crps_samples(samples, y_obs)))
                    _SCORE_CACHE[feat_key] = crps_val
                except Exception as e:
                    logger.debug(f"Prediction failed for {spec.model_id} target {target_month}: {e}")
                    continue

            records.append(
                {
                    "model_id": spec.model_id,
                    "lead": lead,
                    "origin": sp.origin,
                    "target": target_month,
                    "crps": crps_val,
                }
            )

    return records


def score_candidates(
    specs: list[ModelSpec],
    data: dict[str, pd.DataFrame] | pd.DataFrame,
    splitter: Any = None,
    config: Any = None,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Score candidate specifications using expanding-window cross-validation.

    Args:
        specs: List of ModelSpec candidates.
        data: Dictionary of data tables or pre-built training DataFrame.
        splitter: ExpandingWindowSplitter instance (default: 5 years initial, max_lead 3).
        config: DengueCastConfig instance.
        n_jobs: Number of parallel jobs for candidate evaluation.

    Returns:
        DataFrame with columns [model_id, lead, crps_mean, crps].
    """
    if isinstance(data, pd.DataFrame):
        train_df = data.copy()
    elif isinstance(data, dict):
        if "train_df" in data:
            train_df = data["train_df"].copy()
        else:
            train_df, _, _ = build_training_frame(
                cases=data["cases"],
                pop=data["population"],
                climate_obs=data["climate_obs"],
                enso=data["enso"],
                regions=data.get("regions"),
                config=config,
            )
    else:
        raise TypeError(f"Unsupported data format: {type(data)}")

    all_months = sorted(train_df["month"].unique().tolist())

    if splitter is None:
        init_m = min(60, max(24, len(all_months) - 12))
        step_m = max(3, len(all_months) // 10)
        splitter = ExpandingWindowSplitter(initial_train_months=init_m, step=step_m, max_lead=3)

    splits = splitter.split(all_months)
    if not splits:
        logger.warning("No expanding-window splits could be formed with current data size.")
        return pd.DataFrame(columns=["model_id", "lead", "crps_mean", "crps"])

    # Run evaluations (parallel if n_jobs > 1 and multiple specs)
    if n_jobs != 1 and len(specs) > 1:
        nested_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_score_single_spec)(spec, train_df, splits) for spec in specs
        )
        all_records = [r for sublist in nested_results for r in sublist]
    else:
        all_records = []
        for spec in specs:
            all_records.extend(_score_single_spec(spec, train_df, splits))

    if not all_records:
        return pd.DataFrame(columns=["model_id", "lead", "crps_mean", "crps"])

    df = pd.DataFrame(all_records)
    summary = df.groupby(["model_id", "lead"])["crps"].mean().reset_index()
    summary = summary.rename(columns={"crps": "crps_mean"})
    return summary


def select_top_k(
    scores: pd.DataFrame,
    k: int = 4,
    available_specs: list[ModelSpec] | None = None,
) -> list[ModelSpec]:
    """Select top-K candidate models by mean CRPS across leads.

    Always includes baseline models B1 and B2 per MODEL_SPEC C.7.

    Args:
        scores: DataFrame containing columns 'model_id' and 'crps_mean' or 'crps'.
        k: Number of candidate models to select (default: 4).
        available_specs: List of candidate ModelSpec objects to resolve details.

    Returns:
        List of selected ModelSpec objects, including baselines.
    """
    spec_map = {s.model_id: s for s in (available_specs or [])}

    # Baseline specs are always members
    b1_spec = ModelSpec(model_id="B1_seasonal_mean", features=[], description="B1 Seasonal Mean")
    b2_spec = ModelSpec(model_id="B2_seasonal_nb", features=[], description="B2 Seasonal NB")

    selected_specs: list[ModelSpec] = [b1_spec, b2_spec]

    if scores.empty:
        return selected_specs

    # Calculate overall mean CRPS per model
    crps_col = "crps_mean" if "crps_mean" in scores.columns else "crps"
    model_means = scores.groupby("model_id")[crps_col].mean().sort_values(ascending=True)

    # Exclude baselines from candidate ranking
    candidates_ranked = [
        mid for mid in model_means.index if not mid.startswith("B1") and not mid.startswith("B2")
    ]

    top_mids = candidates_ranked[:k]

    for mid in top_mids:
        if mid in spec_map:
            selected_specs.append(spec_map[mid])
        else:
            selected_specs.append(
                ModelSpec(model_id=mid, features=[], description=f"Top candidate {mid}")
            )

    return selected_specs
