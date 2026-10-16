"""CLI entry point for DengueCast.

All sub-commands are registered here via Typer app groups.
Each pipeline stage has its own command:
    denguecast data synth
    denguecast data validate
    denguecast data load
    denguecast data bmkg-load
    denguecast data load-cases
    denguecast data load-indices
    denguecast data compare-climate
    denguecast features build
    denguecast model fit
    denguecast ensemble fit
    denguecast run
    denguecast evaluate
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import typer

from denguecast.logging import setup_logging

app = typer.Typer(
    name="denguecast",
    help="DengueCast — Sistem Peramalan DBD Probabilistik",
    no_args_is_help=True,
)

# Sub-app groups
data_app = typer.Typer(help="Data management commands", no_args_is_help=True)
features_app = typer.Typer(help="Feature engineering commands", no_args_is_help=True)
model_app = typer.Typer(help="Model fitting commands", no_args_is_help=True)
ensemble_app = typer.Typer(help="Ensemble commands", no_args_is_help=True)

app.add_typer(data_app, name="data")
app.add_typer(features_app, name="features")
app.add_typer(model_app, name="model")
app.add_typer(ensemble_app, name="ensemble")


@app.callback()
def main(
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """DengueCast — probabilistic dengue forecasting system."""
    setup_logging(log_level)


# ---------------------------------------------------------------------------
# data commands
# ---------------------------------------------------------------------------


@data_app.command("synth")
def data_synth(
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Generate synthetic data for all pipeline tables."""
    from denguecast.config import load_config
    from denguecast.data.synthetic import generate_synthetic_data

    cfg = load_config(config)
    typer.echo(f"Generating synthetic data with seed={seed}...")
    tables = generate_synthetic_data(
        seed=seed,
        n_regions=cfg.runtime.n_synthetic_regions,
        n_years=cfg.runtime.n_synthetic_years,
        output_dir=cfg.data.paths.processed,
        db_path=cfg.data.paths.db,
    )
    typer.echo(
        f"Successfully generated {len(tables)} tables in '{cfg.data.paths.processed}' and registered in '{cfg.data.paths.db}':"
    )
    for name, df in tables.items():
        typer.echo(f"  - {name}: {len(df)} rows, columns={list(df.columns)}")


@data_app.command("validate")
def data_validate(
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Validate loaded data and print quality report."""
    from denguecast.config import load_config
    from denguecast.data.store import read_table
    from denguecast.data.validate import quality_report

    cfg = load_config(config)
    try:
        cases = read_table("cases", cfg.data.paths.processed)
        climate = read_table("climate_obs", cfg.data.paths.processed)
    except FileNotFoundError as err:
        typer.echo(f"Error loading tables: {err}", err=True)
        raise typer.Exit(code=1) from err

    issues_df, summary = quality_report(cases, climate)
    typer.echo(summary)
    if not issues_df.empty:
        typer.echo("\nSample Issues Found:")
        typer.echo(issues_df.head(10).to_string(index=False))


@data_app.command("load")
def data_load(
    source: str = typer.Option(..., "--source", help="cases | climate | enso | population"),
    path: str = typer.Option(..., "--path"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Load real data via registered adapter."""
    from denguecast.config import load_config
    from denguecast.data.ingest import get_cases_adapter, get_climate_adapter
    from denguecast.data.store import write_table

    cfg = load_config(config)
    s = source.lower()
    if s in ["cases", "case"]:
        adapter = get_cases_adapter("csv")
        df = adapter(path, cfg)
        tbl_name = "cases"
    elif s in ["climate", "climate_obs"]:
        adapter = get_climate_adapter("csv")
        df = adapter(path, cfg)
        tbl_name = "climate_obs"
    else:
        typer.echo(f"Unsupported source type '{source}'.", err=True)
        raise typer.Exit(code=1)

    h = write_table(df, tbl_name, output_dir=cfg.data.paths.processed, db_path=cfg.data.paths.db)
    typer.echo(f"Loaded {len(df)} rows into '{tbl_name}' (data_hash={h[:8]}).")


@data_app.command("bmkg-load")
def data_bmkg_load(
    dir: str = typer.Option("data/raw/bmkg", "--dir"),
    stations: str = typer.Option("data/ref/stations.csv", "--stations"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Load BMKG Data Online station files and aggregate to regions."""
    from pathlib import Path

    import pandas as pd

    from denguecast.config import load_config
    from denguecast.data.ingest.bmkg_dataonline import (
        aggregate_stations_to_regions,
        load_station_file,
        monthly_from_daily,
    )
    from denguecast.data.store import read_table, write_table

    cfg = load_config(config)
    dir_path = Path(dir)
    stn_file = Path(stations)

    if not dir_path.exists():
        typer.echo(f"Directory '{dir_path}' does not exist.", err=True)
        raise typer.Exit(code=1)
    if not stn_file.exists():
        typer.echo(f"Stations reference file '{stn_file}' does not exist.", err=True)
        raise typer.Exit(code=1)

    stations_df = pd.read_csv(stn_file)
    try:
        regions_df = read_table("regions", cfg.data.paths.processed)
    except Exception:
        # Fallback to unique region_id in stations
        regions_df = stations_df[["region_id", "lat", "lon"]].drop_duplicates()

    station_files = (
        list(dir_path.glob("*.csv")) + list(dir_path.glob("*.xlsx")) + list(dir_path.glob("*.xls"))
    )
    if not station_files:
        typer.echo(f"No station files found in '{dir_path}'.", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"Found {len(station_files)} station files. Processing...")
    all_monthly = []
    for sf in station_files:
        # Infer station_id from filename stem or match in stations_df
        sid = sf.stem.split("_")[0]
        try:
            daily = load_station_file(sf, station_id=sid, config=cfg)
            monthly = monthly_from_daily(daily, min_valid_frac=cfg.bmkg.min_valid_days_frac)
            all_monthly.append(monthly)
        except Exception as e:
            typer.echo(f"Warning: Failed to load '{sf.name}': {e}")

    if not all_monthly:
        typer.echo("No valid station data aggregated.", err=True)
        raise typer.Exit(code=1)

    combined_monthly = pd.concat(all_monthly, ignore_index=True)
    clim_obs = aggregate_stations_to_regions(
        combined_monthly,
        stations=stations_df,
        regions=regions_df,
        method=cfg.bmkg.aggregation_method,
        max_km=cfg.bmkg.max_station_km,
        min_stations=cfg.bmkg.min_stations,
    )

    write_table(
        clim_obs, "climate_obs", output_dir=cfg.data.paths.processed, db_path=cfg.data.paths.db
    )
    valid_count = clim_obs["precip"].notna().sum()
    total_count = len(clim_obs)
    pct = (valid_count / total_count * 100.0) if total_count > 0 else 0.0
    typer.echo(f"Successfully aggregated {len(clim_obs)} region-months (source=bmkg_station).")
    typer.echo(
        f"Coverage: {valid_count}/{total_count} ({pct:.1f}%) region-months have valid precip data."
    )


@data_app.command("load-cases")
def data_load_cases(
    source: str = typer.Option("opendengue", "--source", help="opendengue | csv"),
    path: str = typer.Option(..., "--path", help="Path to cases file"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Load dengue case data from OpenDengue or Dinkes CSV."""
    from denguecast.config import load_config
    from denguecast.data.ingest import get_cases_adapter
    from denguecast.data.store import write_table

    cfg = load_config(config)
    adapter = get_cases_adapter(source)
    res = adapter(path, cfg)
    if isinstance(res, tuple):
        cases_df, audit = res
        typer.echo(
            f"Audit: {audit.get('n_provinces')} provinces, {audit.get('total_records')} records."
        )
        if audit.get("unmapped_provinces"):
            typer.echo(f"Warning: Unmapped provinces: {audit['unmapped_provinces']}")
        if audit.get("suspicious_zero_runs"):
            typer.echo(
                f"Warning: Suspicious consecutive zero months in: {audit['suspicious_zero_runs']}"
            )
    else:
        cases_df = res

    h = write_table(
        cases_df, "cases", output_dir=cfg.data.paths.processed, db_path=cfg.data.paths.db
    )
    typer.echo(f"Successfully loaded {len(cases_df)} case records into 'cases' (hash={h[:8]}).")


@data_app.command("load-indices")
def data_load_indices(
    oni: str = typer.Option(..., "--oni", help="Path to ONI data file"),
    dmi: str = typer.Option(..., "--dmi", help="Path to DMI data file"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Load ONI and DMI climate indices."""
    from denguecast.config import load_config
    from denguecast.data.ingest.indices import load_combined
    from denguecast.data.store import write_table

    cfg = load_config(config)
    enso_df = load_combined(oni, dmi)
    h = write_table(enso_df, "enso", output_dir=cfg.data.paths.processed, db_path=cfg.data.paths.db)
    typer.echo(
        f"Successfully loaded {len(enso_df)} climate index months into 'enso' (hash={h[:8]})."
    )


@data_app.command("compare-climate")
def data_compare_climate(
    source_a: str = typer.Option(..., "--source-a", help="First source table name or path"),
    source_b: str = typer.Option(..., "--source-b", help="Second source table name or path"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Compare two climate sources (correlation and bias)."""
    from pathlib import Path

    import pandas as pd

    from denguecast.config import load_config
    from denguecast.data.ingest.era5land import compare_climate
    from denguecast.data.store import read_table

    cfg = load_config(config)

    def _get_df(src: str) -> pd.DataFrame:
        p = Path(src)
        if p.exists():
            return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
        return read_table(src, cfg.data.paths.processed)

    df_a = _get_df(source_a)
    df_b = _get_df(source_b)

    _, summary = compare_climate(df_a, df_b, source_a=source_a, source_b=source_b)
    typer.echo(f"\nClimate Comparison ({summary['source_a']} vs {summary['source_b']}):")
    typer.echo(f"Matched region-months: {summary['n_matched']}")
    typer.echo("-" * 55)
    typer.echo(
        f"{'Variable':<12} | {'Pearson r':<10} | {'Spearman rho':<12} | {'Bias':<8} | {'MAE':<8}"
    )
    typer.echo("-" * 55)
    for var, metrics in summary["metrics"].items():
        typer.echo(
            f"{var:<12} | {metrics['pearson']:<10.3f} | {metrics['spearman']:<12.3f} | {metrics['bias']:<8.2f} | {metrics['mae']:<8.2f}"
        )
    typer.echo("-" * 55)


# ---------------------------------------------------------------------------
# features commands
# ---------------------------------------------------------------------------


@features_app.command("build")
def features_build(
    origin: str = typer.Option(..., "--origin", help="Origin month YYYY-MM"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Build training and forecast feature frames."""
    from denguecast.config import load_config
    from denguecast.data.store import read_table, write_table
    from denguecast.features.build import build_training_frame

    cfg = load_config(config)
    typer.echo(f"Building features with origin={origin}...")

    cases = read_table("cases", cfg.data.paths.processed)
    pop = read_table("population", cfg.data.paths.processed)
    climate = read_table("climate_obs", cfg.data.paths.processed)
    enso = read_table("enso", cfg.data.paths.processed)
    try:
        regions = read_table("regions", cfg.data.paths.processed)
    except Exception:
        regions = None

    train_df, scaler, active_features = build_training_frame(
        cases=cases,
        pop=pop,
        climate_obs=climate,
        enso=enso,
        regions=regions,
        config=cfg,
        origin=origin,
    )

    out_name = f"features_train_{origin.replace('-', '_')}"
    h = write_table(
        train_df, out_name, output_dir=cfg.data.paths.processed, db_path=cfg.data.paths.db
    )

    typer.echo(
        f"Successfully generated {len(train_df)} training rows (table='{out_name}', hash={h[:8]})."
    )
    typer.echo(f"Active features ({len(active_features)}): {active_features}")
    typer.echo(f"Standardized features: {[f'{c}_z' for c in active_features]}")


# ---------------------------------------------------------------------------
# model commands
# ---------------------------------------------------------------------------


@model_app.command("fit")
def model_fit(
    spec: str = typer.Option("nbglm", "--spec", help="baseline | nbglm | b1 | b2"),
    climate_by_group: bool = typer.Option(False, "--climate-by-group"),
    origin: str = typer.Option("2019-12", "--origin", help="Origin month YYYY-MM"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Fit a single model spec on available training data."""
    from denguecast.config import load_config
    from denguecast.data.store import read_table
    from denguecast.features.build import build_training_frame
    from denguecast.models.base import ModelSpec
    from denguecast.models.registry import build_model

    cfg = load_config(config)
    typer.echo(f"Loading data for training up to origin={origin}...")

    cases = read_table("cases", cfg.data.paths.processed)
    pop = read_table("population", cfg.data.paths.processed)
    climate = read_table("climate_obs", cfg.data.paths.processed)
    enso = read_table("enso", cfg.data.paths.processed)
    try:
        regions = read_table("regions", cfg.data.paths.processed)
    except Exception:
        regions = None

    train_df, scaler, active_features = build_training_frame(
        cases=cases,
        pop=pop,
        climate_obs=climate,
        enso=enso,
        regions=regions,
        config=cfg,
        origin=origin,
    )

    typer.echo(
        f"Prepared {len(train_df)} training samples across {train_df['region_id'].nunique()} regions."
    )

    s = spec.lower()
    if s in ["baseline", "baselines"]:
        m1 = build_model(ModelSpec(model_id="B1_seasonal_mean"))
        m1.fit(train_df)
        typer.echo("\n=== B1 Seasonal Expanding Mean ===")
        typer.echo(
            f"Type: {m1.to_dict()['type']}, Region-month cells: {m1.to_dict()['n_region_months']}"
        )

        m2 = build_model(ModelSpec(model_id="B2_seasonal_nb"))
        m2.fit(train_df)
        typer.echo("\n=== B2 Seasonal Negative Binomial ===")
        typer.echo(f"Type: {m2.to_dict()['type']}, Alpha dispersion: {m2.to_dict()['alpha']:.4f}")
        return

    # Select standardized features
    selected_features = [f"{f}_z" for f in active_features if f"{f}_z" in train_df.columns]
    if not selected_features:
        selected_features = [
            f for f in ["humidity_l02", "dtr_l02", "wind_l0"] if f in train_df.columns
        ]

    model_spec = ModelSpec(
        model_id="nbglm",
        features=selected_features,
        climate_by_group=climate_by_group,
        description=f"NBGLM (climate_by_group={climate_by_group})",
    )

    model = build_model(model_spec)
    typer.echo(
        f"Fitting model '{model_spec.model_id}' with {len(selected_features)} features (climate_by_group={climate_by_group})..."
    )
    model.fit(train_df)

    if hasattr(model, "summary_frame"):
        typer.echo("\n=== Model Summary ===")
        summary = model.summary_frame()
        typer.echo(summary.to_string(index=False))

    if hasattr(model, "model_card"):
        typer.echo("\n=== Model Card ===")
        for k, v in model.model_card().items():
            typer.echo(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# ensemble commands
# ---------------------------------------------------------------------------


@ensemble_app.command("fit")
def ensemble_fit(
    origin: str = typer.Option("2019-12", "--origin", help="Origin month YYYY-MM"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Select candidates and fit superensemble (MODEL_SPEC C.7 & C.8)."""
    import uuid

    from denguecast.config import load_config
    from denguecast.data.store import read_table, write_table
    from denguecast.ensemble.candidates import generate_candidates
    from denguecast.ensemble.selection import score_candidates, select_top_k
    from denguecast.ensemble.stacking import Superensemble, compute_weights
    from denguecast.evaluation.scores import crps_samples
    from denguecast.features.build import build_training_frame
    from denguecast.models.base import ModelSpec
    from denguecast.models.registry import build_model

    cfg = load_config(config)
    typer.echo(f"Running ensemble selection and fitting with origin={origin}...")

    cases = read_table("cases", cfg.data.paths.processed)
    pop = read_table("population", cfg.data.paths.processed)
    climate = read_table("climate_obs", cfg.data.paths.processed)
    enso = read_table("enso", cfg.data.paths.processed)
    try:
        regions = read_table("regions", cfg.data.paths.processed)
    except Exception:
        regions = None

    train_df, scaler, active_features = build_training_frame(
        cases=cases,
        pop=pop,
        climate_obs=climate,
        enso=enso,
        regions=regions,
        config=cfg,
        origin=origin,
    )

    typer.echo(f"Training dataset: {len(train_df)} rows, active features: {len(active_features)}")

    # 1. Generate candidate specs
    standardized_features = [f"{f}_z" for f in active_features if f"{f}_z" in train_df.columns]
    all_candidates = generate_candidates(cfg, available_features=standardized_features)
    typer.echo(f"Generated {len(all_candidates)} deterministic candidate specifications.")

    # 2. Score candidates and baselines
    b1_spec = ModelSpec(model_id="B1_seasonal_mean", features=[], description="B1 Seasonal Mean")
    b2_spec = ModelSpec(model_id="B2_seasonal_nb", features=[], description="B2 Seasonal NB")
    eval_specs = [b1_spec, b2_spec] + all_candidates[: min(16, len(all_candidates))]

    typer.echo(f"Scoring {len(eval_specs)} specifications via expanding-window evaluation...")
    scores_df = score_candidates(eval_specs, train_df, config=cfg, n_jobs=4)

    # 3. Select top K candidates plus baselines
    top_k = getattr(cfg.models, "top_k", 4)
    selected_specs = select_top_k(scores_df, k=top_k, available_specs=eval_specs)
    typer.echo(f"\nSelected {len(selected_specs)} models (Top-{top_k} + Baselines):")
    for s in selected_specs:
        typer.echo(f"  - {s.model_id}: {s.features}")

    # 4. Compute weights
    mean_crps = scores_df.groupby("model_id")["crps_mean"].mean()
    # Ensure all selected have a CRPS entry
    for s in selected_specs:
        if s.model_id not in mean_crps:
            mean_crps[s.model_id] = float(mean_crps.mean()) if not mean_crps.empty else 1.0

    selected_mids = [s.model_id for s in selected_specs]
    weights = compute_weights(mean_crps.loc[selected_mids])

    # 5. Fit models and assemble Superensemble
    fitted_models = []
    for s in selected_specs:
        m = build_model(s)
        m.fit(train_df)
        fitted_models.append(m)

    ensemble = Superensemble(fitted_models, weights)

    # 6. Evaluate ensemble vs baseline on latest test data
    test_sub = train_df[train_df["month"] == origin].copy()
    if test_sub.empty:
        test_sub = train_df.tail(63).copy()

    ens_samples = ensemble.predict_samples(test_sub, n_samples=200, seed=42)
    b1_samples = fitted_models[0].predict_samples(test_sub, n_samples=200, seed=42)
    y_test = test_sub["cases"].values

    ens_crps = float(np.mean(crps_samples(ens_samples, y_test)))
    b1_crps = float(np.mean(crps_samples(b1_samples, y_test)))

    typer.echo("\n=== Ensemble Weights ===")
    for mid, w in weights.items():
        crps_val = mean_crps.get(mid, 0.0)
        typer.echo(f"  {mid:<30} | Weight: {w:.4f} | CRPS: {crps_val:.3f}")

    typer.echo("\n=== Performance on Evaluation Subset ===")
    typer.echo(f"  Superensemble CRPS : {ens_crps:.3f}")
    typer.echo(f"  B1 Baseline CRPS   : {b1_crps:.3f}")
    if b1_crps > 0:
        crpss_val = 1.0 - (ens_crps / b1_crps)
        typer.echo(f"  CRPSS vs Baseline  : {crpss_val:.3f}")

    # 7. Record ensemble weights table
    run_id = f"run_{origin.replace('-', '_')}_{uuid.uuid4().hex[:6]}"
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
    h = write_table(
        weights_df,
        "ensemble_weights",
        output_dir=cfg.data.paths.processed,
        db_path=cfg.data.paths.db,
    )
    typer.echo(f"\nWrote ensemble weights to store (data_hash={h[:8]}).")


# ---------------------------------------------------------------------------
# top-level pipeline commands
# ---------------------------------------------------------------------------


@app.command("run")
def run(
    origin: str = typer.Option(..., "--origin", help="Origin month YYYY-MM"),
    synthetic: bool = typer.Option(False, "--synthetic"),
    dry_run: bool = typer.Option(False, "--dry-run"),
    resume: bool = typer.Option(False, "--resume"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Run the full monthly forecast pipeline (MODEL_SPEC C.1 & Prompt 8)."""
    from denguecast.config import load_config
    from denguecast.data.synthetic import generate_synthetic_data
    from denguecast.pipelines.monthly_run import monthly_run

    cfg = load_config(config)
    typer.echo(f"Starting DengueCast monthly run for origin={origin} (dry_run={dry_run})...")

    if synthetic:
        typer.echo("Ensuring synthetic base tables exist...")
        generate_synthetic_data(seed=cfg.runtime.seed)

    result = monthly_run(
        origin=origin,
        config=cfg,
        dry_run=dry_run,
        resume=resume,
        seed=cfg.runtime.seed,
    )

    meta = result["metadata"]
    fc = result["forecasts"]
    typer.echo(f"\nForecast pipeline successfully completed (run_id={meta['run_id']})!")
    typer.echo(f"  Regions evaluated : {meta['n_regions']}")
    typer.echo(f"  Forecast horizons : Lead 1 to {meta['max_lead']}")
    typer.echo(f"  Predictive draws  : {meta['n_samples']} samples per region-lead")
    typer.echo(f"  Total records     : {len(fc)}")
    typer.echo(f"  Selected models   : {', '.join(meta['selected_models'])}")
    if not dry_run:
        typer.echo(f"  Forecasts hash    : {meta.get('forecasts_hash')}")
        typer.echo(f"  Weights hash      : {meta.get('weights_hash')}")


@app.command("evaluate")
def evaluate(
    synthetic: bool = typer.Option(False, "--synthetic"),
    origin: str = typer.Option(None, "--origin"),
    output_report: str = typer.Option("reports/evaluation_report.md", "--output-report"),
    config: str = typer.Option("config/default.yaml", "--config"),
) -> None:
    """Run full expanding-window evaluation, province eligibility gate, and write report."""
    from denguecast.config import load_config
    from denguecast.data.store import read_table, write_table
    from denguecast.data.synthetic import generate_synthetic_data
    from denguecast.evaluation.report import generate_report, run_full_evaluation
    from denguecast.features.build import build_training_frame

    cfg = load_config(config)
    typer.echo(f"Starting DengueCast evaluation (synthetic={synthetic})...")

    try:
        cases = read_table("cases", cfg.data.paths.processed)
        pop = read_table("population", cfg.data.paths.processed)
        climate = read_table("climate_obs", cfg.data.paths.processed)
        enso = read_table("enso", cfg.data.paths.processed)
        try:
            regions = read_table("regions", cfg.data.paths.processed)
        except Exception:
            regions = None
    except FileNotFoundError:
        if synthetic:
            typer.echo("Synthetic tables not found; generating fresh synthetic data...")
            tables = generate_synthetic_data(seed=cfg.runtime.seed)
            cases = tables["cases"]
            pop = tables["population"]
            climate = tables["climate_obs"]
            enso = tables["enso"]
            regions = tables.get("regions")
        else:
            typer.echo("Processed data not found. Run 'denguecast data synth' or load real data first.", err=True)
            raise typer.Exit(code=1) from None

    train_df, _, _ = build_training_frame(
        cases=cases,
        pop=pop,
        climate_obs=climate,
        enso=enso,
        regions=regions,
        config=cfg,
        origin=origin,
    )

    typer.echo(f"Running full expanding-window evaluation on {len(train_df)} observations...")
    scores_df, elig_df = run_full_evaluation(train_df, config=cfg)

    # Sensitivity analysis: without 2020-2021 (COVID-19 impact)
    typer.echo("Running sensitivity evaluation excluding 2020-2021...")
    sens_scores_df, _ = run_full_evaluation(
        train_df,
        config=cfg,
        exclude_periods=["2020", "2021"],
    )

    sens_diff = {}
    if not scores_df.empty:
        sens_diff["Lengkap (Semua Tahun)"] = {
            "crps": float(scores_df["crps"].mean()),
            "crpss": float(scores_df["crpss"].median()),
            "coverage": float(scores_df["coverage_95"].mean()),
        }
    if not sens_scores_df.empty:
        sens_diff["Tanpa 2020-2021 (COVID)"] = {
            "crps": float(sens_scores_df["crps"].mean()),
            "crpss": float(sens_scores_df["crpss"].median()),
            "coverage": float(sens_scores_df["coverage_95"].mean()),
        }

    # Save evaluation tables
    if not scores_df.empty:
        write_table(
            scores_df, "eval_scores", output_dir=cfg.data.paths.processed, db_path=cfg.data.paths.db
        )
    if not elig_df.empty:
        write_table(
            elig_df,
            "province_eligibility",
            output_dir=cfg.data.paths.processed,
            db_path=cfg.data.paths.db,
        )

    # Generate Markdown report
    report_file = generate_report(
        scores_df=scores_df,
        eligibility_df=elig_df,
        output_path=output_report,
        sensitivity_diff=sens_diff,
    )

    typer.echo("\n=== Ringkasan Evaluasi DengueCast ===")
    lead_1_3 = scores_df[scores_df["lead"].isin([1, 2, 3])]
    if not lead_1_3.empty:
        lead_1_3_crpss = float(lead_1_3["crpss"].median())
        lead_1_3_cov = float(lead_1_3["coverage_95"].mean())
        typer.echo(f"  Median CRPSS vs B1 (Lead 1-3) : {lead_1_3_crpss:.3f}")
        typer.echo(f"  Cakupan Interval 95%          : {lead_1_3_cov:.3f}")

    if not elig_df.empty:
        n_layak = int(elig_df["eligible"].sum())
        typer.echo(f"  Gerbang Kelayakan             : {n_layak}/{len(elig_df)} provinsi layak")

    typer.echo(f"\nLaporan evaluasi berhasil ditulis ke '{report_file}'.")


if __name__ == "__main__":
    app()
