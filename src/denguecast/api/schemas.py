"""Pydantic v2 response schemas for DengueCast API (P9)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str = "denguecast"
    service: str = "denguecast"
    version: str = "0.1.0"
    disclaimer: str = (
        "Pemberitahuan: Prototipe riset terinspirasi D-MOSS. "
        "Bukan alat keputusan klinis atau kebijakan. "
        "Kasus berupa DBD berdiagnosis klinis."
    )


class ForecastPoint(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    region_id: str
    origin: str
    target: str
    lead: int = Field(..., ge=1, le=6)
    mean: float
    q025: float
    q50: float
    q975: float
    p_exceed_m1sd: float
    p_exceed_m2sd: float
    p_exceed_p75: float
    p_exceed_p95: float
    risk_level: str
    model_version: str = "0.1.0"
    synthetic: bool = True
    reliable: bool = True


class MapForecastItem(BaseModel):
    region_id: str
    target_month: str
    lead: int
    risk_level: str
    p_exceed: float
    mean: float
    q50: float
    reliable: bool = True
    synthetic: bool = True


class RegionItem(BaseModel):
    region_id: str
    name: str
    region_group: str
    lat: float | None = None
    lon: float | None = None
    reliable: bool = True


class HistoricalCasePoint(BaseModel):
    region_id: str
    month: str
    cases: int
    incidence: float | None = None
    definition: str = "clinical"
    synthetic: bool = True


class EnsembleWeightItem(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    run_id: str
    model_id: str
    features: list[str]
    weight: float
    crps: float | None = None


class RunSummary(BaseModel):
    run_id: str
    origin: str
    completion_time: str
    n_regions: int
    max_lead: int
    synthetic: bool
    selected_models: list[str]


class RunDetail(RunSummary):
    start_time: str
    n_samples: int
    weights: dict[str, float]
    metadata: dict[str, Any] = {}


class EvaluationSummary(BaseModel):
    run_id: str | None = None
    mean_crps: float
    median_crpss: float
    coverage_95: float
    eligible_provinces: int
    total_provinces: int
    disclaimer: str = (
        "Prototipe riset DengueCast. Bukan alat keputusan klinis atau kebijakan. "
        "Kasus berupa DBD berdiagnosis klinis."
    )
