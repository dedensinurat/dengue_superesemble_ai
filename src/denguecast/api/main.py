"""FastAPI application for DengueCast REST API (P9).

Provides read-only access to forecast distributions, run metadata,
regional risk classifications, historical epidemiological series,
and model evaluations.
"""

from __future__ import annotations

import glob
import io
import json
import os
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

from denguecast.api.deps import query_dataframe
from denguecast.api.schemas import (
    EnsembleWeightItem,
    EvaluationSummary,
    ForecastPoint,
    HealthResponse,
    HistoricalCasePoint,
    MapForecastItem,
    RegionItem,
    RunDetail,
    RunSummary,
)

app = FastAPI(
    title="DengueCast API",
    description=(
        "Sistem Peramalan DBD Probabilistik - REST API\n\n"
        "**Pemberitahuan**: Prototipe riset terinspirasi D-MOSS. "
        "Bukan alat keputusan klinis atau kebijakan. "
        "Kasus berupa DBD berdiagnosis klinis."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health() -> HealthResponse:
    """Service health check."""
    return HealthResponse()


@app.get("/v1/runs", response_model=list[RunSummary], tags=["Forecasts"])
async def list_runs() -> list[RunSummary]:
    """List available forecast execution runs."""
    summary_files = glob.glob("reports/runs/*.json")
    runs: list[RunSummary] = []

    for fpath in sorted(summary_files, reverse=True):
        try:
            with open(fpath, encoding="utf-8") as f:
                d = json.load(f)
            runs.append(
                RunSummary(
                    run_id=d.get("run_id", os.path.basename(fpath).replace(".json", "")),
                    origin=d.get("origin", ""),
                    completion_time=d.get("completion_time", ""),
                    n_regions=d.get("n_regions", 63),
                    max_lead=d.get("max_lead", 6),
                    synthetic=d.get("synthetic", True),
                    selected_models=d.get("selected_models", []),
                )
            )
        except Exception:
            pass

    if not runs:
        # Fallback to querying distinct run_id from forecasts table
        fc = query_dataframe("forecasts")
        if not fc.empty and "run_id" in fc.columns:
            for rid, grp in fc.groupby("run_id"):
                runs.append(
                    RunSummary(
                        run_id=str(rid),
                        origin=str(grp["origin"].iloc[0]),
                        completion_time="N/A",
                        n_regions=int(grp["region_id"].nunique()),
                        max_lead=int(grp["lead"].max()),
                        synthetic=bool(grp["synthetic"].iloc[0]),
                        selected_models=[],
                    )
                )

    return runs


@app.get("/v1/runs/{run_id}", response_model=RunDetail, tags=["Forecasts"])
async def get_run_detail(run_id: str) -> RunDetail:
    """Get detailed configuration and model weights for a specific run."""
    summary_files = glob.glob(f"reports/runs/*{run_id}*.json")
    if summary_files:
        with open(summary_files[0], encoding="utf-8") as f:
            d = json.load(f)
        return RunDetail(
            run_id=d.get("run_id", run_id),
            origin=d.get("origin", ""),
            completion_time=d.get("completion_time", ""),
            n_regions=d.get("n_regions", 63),
            max_lead=d.get("max_lead", 6),
            synthetic=d.get("synthetic", True),
            selected_models=d.get("selected_models", []),
            start_time=d.get("start_time", ""),
            n_samples=d.get("n_samples", 1000),
            weights=d.get("weights", {}),
            metadata=d,
        )

    # Fallback to table
    wt = query_dataframe("ensemble_weights", f"run_id = '{run_id}'")
    if wt.empty:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")

    weights_dict = {str(r["model_id"]): float(r["weight"]) for _, r in wt.iterrows()}
    return RunDetail(
        run_id=run_id,
        origin="N/A",
        completion_time="N/A",
        n_regions=63,
        max_lead=6,
        synthetic=True,
        selected_models=list(weights_dict.keys()),
        start_time="N/A",
        n_samples=1000,
        weights=weights_dict,
    )


@app.get("/v1/regions", response_model=list[RegionItem], tags=["Metadata"])
async def list_regions() -> list[RegionItem]:
    """List administrative regions with geographic coordinates and eligibility status."""
    regs = query_dataframe("regions")
    elig = query_dataframe("province_eligibility")
    elig_map = {str(r["region_id"]): bool(r["eligible"]) for _, r in elig.iterrows()} if not elig.empty else {}

    items: list[RegionItem] = []
    if not regs.empty:
        for _, r in regs.iterrows():
            rid = str(r["region_id"])
            items.append(
                RegionItem(
                    region_id=rid,
                    name=str(r.get("name", rid)),
                    region_group=str(r.get("region_group", "Indonesia")),
                    lat=float(r["lat"]) if "lat" in r and pd.notna(r["lat"]) else None,
                    lon=float(r["lon"]) if "lon" in r and pd.notna(r["lon"]) else None,
                    reliable=elig_map.get(rid, True),
                )
            )
    else:
        # Fallback from cases
        cases = query_dataframe("cases")
        if not cases.empty:
            for rid in sorted(cases["region_id"].unique()):
                items.append(
                    RegionItem(
                        region_id=str(rid),
                        name=f"Wilayah {rid}",
                        region_group="Indonesia",
                        reliable=elig_map.get(str(rid), True),
                    )
                )

    return items


@app.get("/v1/forecasts", response_model=list[ForecastPoint], tags=["Forecasts"])
async def get_forecasts(
    run_id: str | None = None,
    region_id: str | None = None,
    lead: int | None = Query(None, ge=1, le=6),
    target_month: str | None = None,
) -> list[ForecastPoint]:
    """Retrieve probabilistic forecasts filtered by region, lead, or target month."""
    fc = query_dataframe("forecasts")
    if fc.empty:
        return []

    # Apply filters
    if run_id:
        fc = fc[fc["run_id"] == run_id]
    if region_id:
        fc = fc[fc["region_id"] == region_id]
    if lead:
        fc = fc[fc["lead"] == lead]
    if target_month:
        fc = fc[fc["target"] == target_month]

    results: list[ForecastPoint] = []
    for _, r in fc.iterrows():
        results.append(
            ForecastPoint(
                run_id=str(r["run_id"]),
                region_id=str(r["region_id"]),
                origin=str(r["origin"]),
                target=str(r["target"]),
                lead=int(r["lead"]),
                mean=float(r["mean"]),
                q025=float(r["q025"]),
                q50=float(r["q50"]),
                q975=float(r["q975"]),
                p_exceed_m1sd=float(r["p_exceed_m1sd"]),
                p_exceed_m2sd=float(r["p_exceed_m2sd"]),
                p_exceed_p75=float(r["p_exceed_p75"]),
                p_exceed_p95=float(r["p_exceed_p95"]),
                risk_level=str(r["risk_level"]),
                model_version=str(r.get("model_version", "0.1.0")),
                synthetic=bool(r.get("synthetic", True)),
                reliable=bool(r.get("reliable", True)),
            )
        )
    return results


@app.get("/v1/forecasts/map", response_model=list[MapForecastItem], tags=["Forecasts"])
async def get_forecasts_map(
    run_id: str | None = None,
    target_month: str | None = None,
    lead: int = Query(1, ge=1, le=6),
    threshold: Literal["m1sd", "m2sd", "p75", "p95"] = "m1sd",
) -> list[MapForecastItem]:
    """Map-oriented summary of risk level and exceedance probability per region."""
    fc = query_dataframe("forecasts")
    if fc.empty:
        return []

    if run_id:
        fc = fc[fc["run_id"] == run_id]
    fc = fc[fc["target"] == target_month] if target_month else fc[fc["lead"] == lead]

    col_map = {
        "m1sd": "p_exceed_m1sd",
        "m2sd": "p_exceed_m2sd",
        "p75": "p_exceed_p75",
        "p95": "p_exceed_p95",
    }
    p_col = col_map[threshold]

    items: list[MapForecastItem] = []
    for _, r in fc.iterrows():
        items.append(
            MapForecastItem(
                region_id=str(r["region_id"]),
                target_month=str(r["target"]),
                lead=int(r["lead"]),
                risk_level=str(r["risk_level"]),
                p_exceed=float(r.get(p_col, 0.0)),
                mean=float(r["mean"]),
                q50=float(r["q50"]),
                reliable=bool(r.get("reliable", True)),
                synthetic=bool(r.get("synthetic", True)),
            )
        )
    return items


@app.get("/v1/history", response_model=list[HistoricalCasePoint], tags=["Epidemiology"])
async def get_history(
    region_id: str | None = None,
    last_n_months: int = Query(24, ge=1, le=240),
) -> list[HistoricalCasePoint]:
    """Retrieve historical dengue case time series."""
    cases = query_dataframe("cases")
    if cases.empty:
        return []

    if region_id:
        cases = cases[cases["region_id"] == region_id]

    cases = cases.sort_values("month", ascending=False)
    unique_months = cases["month"].unique()[:last_n_months]
    cases_sub = cases[cases["month"].isin(unique_months)].sort_values(["region_id", "month"])

    records: list[HistoricalCasePoint] = []
    for _, r in cases_sub.iterrows():
        records.append(
            HistoricalCasePoint(
                region_id=str(r["region_id"]),
                month=str(r["month"]),
                cases=int(r["cases"]),
                definition=str(r.get("definition", "clinical")),
                synthetic=bool(r.get("synthetic", True)),
            )
        )
    return records


@app.get("/v1/evaluation/summary", response_model=EvaluationSummary, tags=["Evaluation"])
async def get_evaluation_summary(run_id: str | None = None) -> EvaluationSummary:
    """Summary of expanding-window verification metrics."""
    scores = query_dataframe("eval_scores")
    elig = query_dataframe("province_eligibility")

    if scores.empty:
        return EvaluationSummary(
            mean_crps=0.0,
            median_crpss=0.0,
            coverage_95=0.0,
            eligible_provinces=0,
            total_provinces=0,
        )

    if run_id:
        scores = scores[scores["run_id"] == run_id]

    mean_crps = float(scores["crps"].mean()) if "crps" in scores.columns else 0.0
    median_crpss = float(scores["crpss"].median()) if "crpss" in scores.columns else 0.0
    cov_95 = float(scores["coverage_95"].mean()) if "coverage_95" in scores.columns else 0.0

    n_elig = int(elig["eligible"].sum()) if not elig.empty else 0
    total_prov = len(elig) if not elig.empty else 63

    return EvaluationSummary(
        run_id=run_id,
        mean_crps=round(mean_crps, 3),
        median_crpss=round(median_crpss, 3),
        coverage_95=round(cov_95, 3),
        eligible_provinces=n_elig,
        total_provinces=total_prov,
    )


@app.get("/v1/weights", response_model=list[EnsembleWeightItem], tags=["Models"])
async def get_ensemble_weights(run_id: str | None = None) -> list[EnsembleWeightItem]:
    """Retrieve superensemble model weights."""
    wt = query_dataframe("ensemble_weights")
    if wt.empty:
        return []

    if run_id:
        wt = wt[wt["run_id"] == run_id]

    items: list[EnsembleWeightItem] = []
    for _, r in wt.iterrows():
        feat_str = str(r.get("features", ""))
        items.append(
            EnsembleWeightItem(
                run_id=str(r["run_id"]),
                model_id=str(r["model_id"]),
                features=feat_str.split(",") if feat_str else [],
                weight=float(r["weight"]),
                crps=float(r["crps"]) if "crps" in r and pd.notna(r["crps"]) else None,
            )
        )
    return items


@app.get("/v1/download/forecasts.csv", tags=["Export"])
async def download_forecasts_csv(run_id: str | None = None) -> Response:
    """Download forecasts table as CSV file."""
    fc = query_dataframe("forecasts")
    if fc.empty:
        raise HTTPException(status_code=404, detail="No forecasts available.")

    if run_id:
        fc = fc[fc["run_id"] == run_id]

    csv_buf = io.StringIO()
    fc.to_csv(csv_buf, index=False)
    filename = f"forecasts_{run_id or 'all'}.csv"

    return Response(
        content=csv_buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
