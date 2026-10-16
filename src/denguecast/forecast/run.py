"""Main forecast runner for DengueCast monthly pipeline (P8).

Executes the end-to-end forecasting sequence:
1. Anti-leakage feature preparation (observations strictly <= origin).
2. Endemic channel baseline computation.
3. Recursive lagged case reconstruction (M0 for leads h=2..6).
4. Superensemble candidate selection and fitting.
5. Future climate provisioning (ClimatologyProvider / SyntheticEnsembleProvider).
6. Multi-step ahead probabilistic simulation (1000 draws).
7. Risk level categorization and exceedance probability calculation.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from denguecast.climate.climatology import ClimatologyProvider
from denguecast.ensemble.candidates import generate_candidates
from denguecast.ensemble.selection import score_candidates, select_top_k
from denguecast.ensemble.stacking import Superensemble, compute_weights
from denguecast.features.build import build_forecast_frame, build_training_frame
from denguecast.features.thresholds import endemic_channel
from denguecast.forecast.outputs import format_forecasts
from denguecast.models.base import ModelSpec
from denguecast.models.recursive_cases import ReconstructLag
from denguecast.models.registry import build_model

logger = logging.getLogger(__name__)


def run_forecast(
    origin: str,
    data: dict[str, pd.DataFrame],
    config: Any = None,
    dry_run: bool = False,
    n_samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute complete forecast pipeline for given origin month.

    Args:
        origin: Origin month string 'YYYY-MM'.
        data: Dict of data tables ('cases', 'population', 'climate_obs', 'enso', 'regions').
        config: DengueCastConfig instance.
        dry_run: If True, executes pipeline logic without writing to disk.
        n_samples: Number of probabilistic predictive draws (default: 1000).
        seed: Random seed for reproducibility.

    Returns:
        Dict with 'run_id', 'forecasts', 'weights', 'metadata'.
    """
    run_id = f"run_{origin.replace('-', '_')}_{uuid.uuid4().hex[:6]}"
    start_time = datetime.now(UTC).isoformat()

    cases = data["cases"].copy()
    population = data["population"].copy()
    climate_obs = data["climate_obs"].copy()
    enso = data["enso"].copy()
    regions = data.get("regions", pd.DataFrame()).copy()

    # Validate origin exists in cases
    available_months = sorted(cases["month"].unique())
    if origin not in available_months:
        raise ValueError(
            f"Origin month '{origin}' not found in cases data. Available range: "
            f"{available_months[0]} to {available_months[-1]}."
        )

    # 1. Build training matrix strictly up to origin (anti-leakage)
    train_df, scaler, active_features = build_training_frame(
        cases=cases,
        pop=population,
        climate_obs=climate_obs,
        enso=enso,
        regions=regions if not regions.empty else None,
        config=config,
        origin=origin,
    )

    # 2. Compute endemic channels from historical incidence (5 years before origin)
    cases_hist = cases[cases["month"] <= origin].copy()
    cases_hist["year"] = cases_hist["month"].str.slice(0, 4).astype(int)
    pop_lookup_df = population[["region_id", "year", "population"]].copy()
    merged_inc = pd.merge(cases_hist, pop_lookup_df, on=["region_id", "year"], how="left")
    merged_inc["population"] = merged_inc["population"].fillna(1_000_000).astype(int)
    merged_inc["incidence"] = 100_000.0 * merged_inc["cases"] / merged_inc["population"]

    channel_df = endemic_channel(merged_inc, origin=origin, years=5)

    # 3. Reconstruct lagged cases for h=1..6
    max_lead = 6
    if config and hasattr(config, "horizon"):
        max_lead = getattr(config.horizon, "max_lead", 6)

    reconstructor = ReconstructLag().fit(train_df)
    ylag_table = reconstructor.reconstruct(
        origin=origin,
        max_lead=max_lead,
        cases_df=cases,
        pop_df=population,
        regions_df=regions if not regions.empty else None,
    )

    # 4. Candidate selection and Superensemble fitting
    standardized_feats = [f"{f}_z" for f in active_features if f"{f}_z" in train_df.columns]
    all_candidates = generate_candidates(config, available_features=standardized_feats)

    b1_spec = ModelSpec(model_id="B1_seasonal_mean", features=[], description="B1 Seasonal Mean")
    b2_spec = ModelSpec(model_id="B2_seasonal_nb", features=[], description="B2 Seasonal NB")
    eval_specs = [b1_spec, b2_spec] + all_candidates[: min(12, len(all_candidates))]

    scores_df = score_candidates(eval_specs, train_df, config=config, n_jobs=4)
    top_k = getattr(config.models, "top_k", 4) if config and hasattr(config, "models") else 4
    selected_specs = select_top_k(scores_df, k=top_k, available_specs=eval_specs)

    # Calculate weights
    mean_crps = (
        scores_df.groupby("model_id")["crps_mean"].mean()
        if not scores_df.empty
        else pd.Series()
    )
    for s in selected_specs:
        if s.model_id not in mean_crps:
            mean_crps[s.model_id] = 1.0

    selected_mids = [s.model_id for s in selected_specs]
    weights = compute_weights(mean_crps.loc[selected_mids])

    # Fit selected models on all available training data <= origin
    fitted_models = []
    for s in selected_specs:
        m = build_model(s)
        m.fit(train_df)
        fitted_models.append(m)

    ensemble = Superensemble(fitted_models, weights)

    # 5. Future Climate Provider (ClimatologyProvider for Stage A)
    provider = ClimatologyProvider(climate_obs=climate_obs, enso_obs=enso, window_years=10)

    # 6. Generate forecasts for all regions and leads h=1..max_lead
    samples_by_region_lead: dict[tuple[str, int], np.ndarray] = {}

    pop_lookup = population.set_index(["region_id", "year"])["population"].to_dict()

    for h in range(1, max_lead + 1):
        ylag_sub = ylag_table[ylag_table["lead"] == h]
        forecast_frame = build_forecast_frame(
            origin=origin,
            lead=h,
            provider=provider,
            ylag_val=ylag_sub,
            climate_obs=climate_obs,
            enso_obs=enso,
            pop=population,
            regions=regions,
            config=config,
            scaler=scaler,
        )

        samples = ensemble.predict_samples(
            forecast_frame, n_samples=n_samples, seed=seed + h * 31
        )

        for row_idx, row in forecast_frame.iterrows():
            rid = str(row["region_id"])
            samples_by_region_lead[(rid, h)] = samples[row_idx, :]

    # 7. Check province eligibility if available
    elig_lookup = {}
    if "province_eligibility" in data:
        for _, r in data["province_eligibility"].iterrows():
            elig_lookup[str(r["region_id"])] = bool(r["eligible"])

    # 8. Format forecasts table
    is_synthetic = bool(cases.get("synthetic", pd.Series([True])).iloc[0])
    forecasts_df = format_forecasts(
        samples_by_region_lead=samples_by_region_lead,
        origin=origin,
        channel_df=channel_df,
        pop_lookup=pop_lookup,
        run_id=run_id,
        config=config,
        eligibility_lookup=elig_lookup,
        model_version="0.1.0",
        synthetic=is_synthetic,
    )

    # 9. Format ensemble weights table
    weight_rows = []
    for s in selected_specs:
        weight_rows.append(
            {
                "run_id": run_id,
                "model_id": s.model_id,
                "features": ",".join(s.features),
                "weight": float(weights.get(s.model_id, 0.0)),
                "crps": float(mean_crps.get(s.model_id, 0.0)),
            }
        )
    weights_df = pd.DataFrame(weight_rows)

    metadata = {
        "run_id": run_id,
        "origin": origin,
        "start_time": start_time,
        "completion_time": datetime.now(UTC).isoformat(),
        "n_regions": forecasts_df["region_id"].nunique(),
        "max_lead": max_lead,
        "n_samples": n_samples,
        "synthetic": is_synthetic,
        "selected_models": [s.model_id for s in selected_specs],
        "weights": weights.to_dict(),
    }

    return {
        "run_id": run_id,
        "forecasts": forecasts_df,
        "weights": weights_df,
        "metadata": metadata,
    }
