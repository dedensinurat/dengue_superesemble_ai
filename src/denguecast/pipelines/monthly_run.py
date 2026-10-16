"""Monthly forecast pipeline (MODEL_SPEC C.1 & Prompt 8).

Complete end-to-end execution pipeline triggered via:
    denguecast run --origin YYYY-MM
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from denguecast.config import load_config
from denguecast.data.store import read_table, write_table
from denguecast.data.validate import flag_possibly_incomplete, quality_report
from denguecast.forecast.run import run_forecast

logger = logging.getLogger(__name__)


def monthly_run(
    origin: str,
    config: Any = None,
    dry_run: bool = False,
    resume: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the complete monthly DengueCast pipeline.

    Steps:
    1. Load and validate data; flag latest reporting delays.
    2. Feature engineering & recursive lag reconstruction (h=2..6).
    3. Superensemble candidate selection, weighting, and fitting.
    4. Probabilistic simulation for leads h=1..6 across all regions.
    5. Persistence: writes 'forecasts' and 'ensemble_weights' to DuckDB/Parquet.
    6. Writes run summary JSON.

    Args:
        origin: Forecast origin month 'YYYY-MM'.
        config: Config path or DengueCastConfig instance.
        dry_run: If True, executes pipeline without writing to disk.
        resume: If True, reuses existing run outputs if already completed.
        seed: Random seed.

    Returns:
        Summary dict containing run_id and metadata.
    """
    if config is None or isinstance(config, str):
        cfg = load_config(config or "config/default.yaml")
    else:
        cfg = config

    logger.info(f"=== Starting DengueCast Monthly Pipeline (origin={origin}, dry_run={dry_run}) ===")

    # 1. Load data tables
    cases = read_table("cases", cfg.data.paths.processed)
    pop = read_table("population", cfg.data.paths.processed)
    climate = read_table("climate_obs", cfg.data.paths.processed)
    enso = read_table("enso", cfg.data.paths.processed)
    try:
        regions = read_table("regions", cfg.data.paths.processed)
    except Exception:
        regions = None

    try:
        elig_table = read_table("province_eligibility", cfg.data.paths.processed)
    except Exception:
        elig_table = None

    data_bundle = {
        "cases": cases,
        "population": pop,
        "climate_obs": climate,
        "enso": enso,
        "regions": regions,
    }
    if elig_table is not None:
        data_bundle["province_eligibility"] = elig_table

    # Data validation
    cases_flagged = flag_possibly_incomplete(cases)
    issues_df, quality_summary = quality_report(cases_flagged, climate)
    logger.info(f"Data validation completed:\n{quality_summary}")

    # 2. Check if output already exists (idempotency and --resume)
    run_summary_dir = "reports/runs"
    os.makedirs(run_summary_dir, exist_ok=True)
    summary_file = os.path.join(run_summary_dir, f"run_{origin.replace('-', '_')}.json")

    if resume and os.path.exists(summary_file):
        logger.info(f"Run for origin {origin} already exists; resuming by reading cached summary.")
        with open(summary_file, encoding="utf-8") as f:
            return json.load(f)

    # 3. Run forecast engine
    result = run_forecast(
        origin=origin,
        data=data_bundle,
        config=cfg,
        dry_run=dry_run,
        n_samples=cfg.models.n_samples if hasattr(cfg, "models") else 1000,
        seed=seed,
    )

    forecasts_df = result["forecasts"]
    weights_df = result["weights"]
    metadata = result["metadata"]

    # 4. Persistence (DuckDB and Parquet)
    if not dry_run:
        h_fc = write_table(
            forecasts_df,
            "forecasts",
            output_dir=cfg.data.paths.processed,
            db_path=cfg.data.paths.db,
        )
        h_wt = write_table(
            weights_df,
            "ensemble_weights",
            output_dir=cfg.data.paths.processed,
            db_path=cfg.data.paths.db,
        )
        metadata["forecasts_hash"] = h_fc[:8]
        metadata["weights_hash"] = h_wt[:8]

        # Save summary JSON
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(
            f"Successfully stored {len(forecasts_df)} forecast records into 'forecasts' and "
            f"'{summary_file}'."
        )

    logger.info(
        f"=== DengueCast Monthly Pipeline Completed for {origin}: "
        f"{metadata['n_regions']} regions, {metadata['max_lead']} leads ==="
    )
    return result
