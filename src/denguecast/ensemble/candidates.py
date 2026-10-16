"""Generate candidate model specifications (MODEL_SPEC C.7).

Enumerates subsets of environmental predictor groups:
- Temperature: {tmean} or {tmin, tmax}
- Diurnal temperature range: {dtr}
- Humidity: {humidity_l02} or {humidity_l04}
- Precipitation: {precip_l02} or {precip_l04}
- Wind: {wind_l0}
- Teleconnections (ENSO/IOD): {oni_l03} or {nino34_l03} or {dmi_l03}
- Landcover: {urban_pct, periurban_pct}

Rules:
- Mutual exclusion: tmin and tmax must never appear with tmean or dtr.
- Collinearity: maximum features per specification.
- Deterministic lexicographical ordering.
- Hard limit on candidate count (default: 128).
"""

from __future__ import annotations

import itertools
from typing import Any

from denguecast.models.base import ModelSpec


def generate_candidates(
    config: Any = None,
    available_features: list[str] | None = None,
) -> list[ModelSpec]:
    """Enumerate all valid candidate model specifications.

    Args:
        config: DengueCastConfig or dict-like configuration.
        available_features: Optional explicit list of feature column names available in data.

    Returns:
        Deterministic list of ModelSpec instances (up to max_candidates).
    """
    # Extract settings from config
    max_candidates = 128
    climate_by_group = False

    if config is not None:
        if hasattr(config, "models"):
            max_candidates = getattr(config.models, "max_candidates", 128)
            climate_by_group = getattr(config.models, "climate_by_group", False)
        elif isinstance(config, dict):
            max_candidates = config.get("models", {}).get("max_candidates", 128)
            climate_by_group = config.get("models", {}).get("climate_by_group", False)

    # Standard feature candidates
    # Group definitions (standardized or raw)
    # 1. Temperature groups (mutually exclusive)
    temp_singles = ["tmean_l02_z", "tmean_l02"]
    temp_duals = [("tmin_l02_z", "tmax_l02_z"), ("tmin_l02", "tmax_l02")]

    # 2. DTR group
    dtr_options = ["dtr_l02_z", "dtr_l02"]

    # 3. Humidity options
    hum_options = ["humidity_l02_z", "humidity_l04_z", "humidity_l02", "humidity_l04"]

    # 4. Precipitation options
    precip_options = ["precip_l02_z", "precip_l04_z", "precip_l02", "precip_l04"]

    # 5. Wind options
    wind_options = ["wind_l0_z", "wind_l0"]

    # 6. Index options
    index_options = [
        "oni_l03_z",
        "dmi_l03_z",
        "nino34_l03_z",
        "oni_l03",
        "dmi_l03",
        "nino34_l03",
    ]

    # Filter by available_features if provided
    if available_features:
        avail_set = set(available_features)
        temp_singles = [f for f in temp_singles if f in avail_set]
        temp_duals = [pair for pair in temp_duals if pair[0] in avail_set and pair[1] in avail_set]
        dtr_options = [f for f in dtr_options if f in avail_set]
        hum_options = [f for f in hum_options if f in avail_set]
        precip_options = [f for f in precip_options if f in avail_set]
        wind_options = [f for f in wind_options if f in avail_set]
        index_options = [f for f in index_options if f in avail_set]

    # Choose canonical choices for clean grid
    t_single = temp_singles[:1]
    t_dual = [temp_duals[0]] if temp_duals else []
    dtr_ch = dtr_options[:1]
    hum_ch = hum_options[:2]
    prec_ch = precip_options[:2]
    wind_ch = wind_options[:1]
    idx_ch = index_options[:3]

    candidate_specs: list[tuple[str, ...]] = []

    # Simple core models
    # Level 1: 1-2 predictors
    for h in hum_ch:
        candidate_specs.append((h,))
    for p in prec_ch:
        candidate_specs.append((p,))
    for h, p in itertools.product(hum_ch, prec_ch):
        candidate_specs.append((h, p))
    for h, d in itertools.product(hum_ch, dtr_ch):
        candidate_specs.append((h, d))

    # Level 2: 3-predictor models
    for h, p, d in itertools.product(hum_ch, prec_ch, dtr_ch):
        candidate_specs.append((h, p, d))
    for h, d, idx in itertools.product(hum_ch, dtr_ch, idx_ch):
        candidate_specs.append((h, d, idx))
    for h, p, idx in itertools.product(hum_ch, prec_ch, idx_ch):
        candidate_specs.append((h, p, idx))
    for p, d, idx in itertools.product(prec_ch, dtr_ch, idx_ch):
        candidate_specs.append((p, d, idx))

    # Level 3: 4-predictor models
    for h, p, d, idx in itertools.product(hum_ch, prec_ch, dtr_ch, idx_ch):
        candidate_specs.append((h, p, d, idx))
    for h, p, d, w in itertools.product(hum_ch, prec_ch, dtr_ch, wind_ch):
        candidate_specs.append((h, p, d, w))
    for h, p, idx, w in itertools.product(hum_ch, prec_ch, idx_ch, wind_ch):
        candidate_specs.append((h, p, idx, w))

    # Level 4: Temperature models (with mutual exclusion)
    # With tmean:
    for tm in t_single:
        for h in hum_ch:
            candidate_specs.append((tm, h))
        for h, p in itertools.product(hum_ch, prec_ch):
            candidate_specs.append((tm, h, p))
        for h, p, idx in itertools.product(hum_ch, prec_ch, idx_ch):
            candidate_specs.append((tm, h, p, idx))

    # With tmin + tmax (never with tmean or dtr):
    for td in t_dual:
        for h in hum_ch:
            candidate_specs.append((td[0], td[1], h))
        for p in prec_ch:
            candidate_specs.append((td[0], td[1], p))
        for h, p in itertools.product(hum_ch, prec_ch):
            candidate_specs.append((td[0], td[1], h, p))
        for h, idx in itertools.product(hum_ch, idx_ch):
            candidate_specs.append((td[0], td[1], h, idx))
        for h, p, idx in itertools.product(hum_ch, prec_ch, idx_ch):
            candidate_specs.append((td[0], td[1], h, p, idx))

    # Deduplicate and sort deterministically
    unique_tuples = sorted(list(set(candidate_specs)))

    # Convert to ModelSpec
    specs: list[ModelSpec] = []
    for i, feats_tuple in enumerate(unique_tuples):
        feats = list(feats_tuple)
        model_id = f"nbglm_{i + 1:03d}_{'_'.join(feats[:2])}"
        specs.append(
            ModelSpec(
                model_id=model_id,
                features=feats,
                climate_by_group=climate_by_group,
                description=f"NBGLM candidate {i + 1:03d} with {len(feats)} features",
            )
        )
        if len(specs) >= max_candidates:
            break

    return specs
