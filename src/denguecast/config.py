"""Configuration management for DengueCast.

Loads config/default.yaml and allows override via environment variables.
All runtime parameters come from here — no magic numbers in code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RegionConfig(BaseModel):
    """Region/administrative unit configuration."""

    country: str = "Indonesia"
    admin_level: int = 1
    id_col: str = "region_id"


class DataPaths(BaseModel):
    """File paths for data storage."""

    raw: str = "data/raw"
    interim: str = "data/interim"
    processed: str = "data/processed"
    db: str = "data/processed/denguecast.duckdb"


class DataConfig(BaseModel):
    """Data source and path configuration."""

    paths: DataPaths = Field(default_factory=DataPaths)
    month_col: str = "month"
    cases_source: str = "synthetic"
    climate_source: str = "climatology"


class HorizonConfig(BaseModel):
    """Forecast horizon configuration."""

    max_lead: int = 6


class FeaturesConfig(BaseModel):
    """Feature engineering configuration."""

    climate_window: int = 3
    nino_window: int = 4
    collinearity_threshold: float = 0.7
    climate_by_group: bool = False
    exclude_dtr_if_missing: bool = True


class ModelsConfig(BaseModel):
    """Model and ensemble configuration."""

    n_samples: int = 1000
    max_candidates: int = 128
    top_k: int = 4
    propagate_lag_uncertainty: bool = False


class EvaluationConfig(BaseModel):
    """Evaluation and backtesting configuration."""

    initial_train_years: int = 5
    cutoffs: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75])
    exclude_periods: list[str] = Field(default_factory=lambda: ["2020-01/2021-12"])
    bootstrap_n: int = 500
    reliability_bins: int = 10


class ThresholdsConfig(BaseModel):
    """Epidemic threshold (endemic channel) configuration."""

    endemic_years: int = 5
    exclude_outbreak_years: bool = False


class BmkgConfig(BaseModel):
    """BMKG Data Online adapter configuration."""

    column_map: dict[str, str] = Field(
        default_factory=lambda: {
            "Tn": "tmin",
            "Tx": "tmax",
            "Tavg": "tmean",
            "RH_avg": "humidity",
            "RR": "precip",
            "ss": "sunshine",
            "ff_x": "wind_max",
            "ddd_x": "wind_dir_max",
            "ff_avg": "wind",
            "ddd_car": "wind_dir_mode",
        }
    )
    date_formats: list[str] = Field(default_factory=lambda: ["%d-%m-%Y", "%Y-%m-%d"])
    codes: dict[int, str] = Field(default_factory=lambda: {8888: "nan", 9999: "nan"})
    min_valid_days_frac: float = 0.8
    aggregation_method: str = "idw"
    max_station_km: float = 150.0
    min_stations: int = 1
    georef_rmse_max_km: float = 50.0


class RuntimeConfig(BaseModel):
    """Runtime and reproducibility settings."""

    seed: int = 42
    synthetic: bool = True
    n_synthetic_regions: int = 63
    n_synthetic_years: int = 12
    log_level: str = "INFO"


class ApiConfig(BaseModel):
    """FastAPI server settings."""

    cors_origins: list[str] = Field(default_factory=lambda: ["*"])
    host: str = "0.0.0.0"
    port: int = 8000


class DashboardConfig(BaseModel):
    """Streamlit dashboard settings."""

    port: int = 8501


class DengueCastConfig(BaseModel):
    """Root configuration model for DengueCast."""

    region: RegionConfig = Field(default_factory=RegionConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    horizon: HorizonConfig = Field(default_factory=HorizonConfig)
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    thresholds: ThresholdsConfig = Field(default_factory=ThresholdsConfig)
    bmkg: BmkgConfig = Field(default_factory=BmkgConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)


def load_config(path: str | Path | None = None) -> DengueCastConfig:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML config file. Defaults to config/default.yaml.

    Returns:
        Validated DengueCastConfig instance.
    """
    if path is None:
        path = Path(__file__).parent.parent.parent / "config" / "default.yaml"

    config_path = Path(path)
    if not config_path.exists():
        return DengueCastConfig()

    with open(config_path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return DengueCastConfig.model_validate(raw)


# Module-level default config singleton
_default_config: DengueCastConfig | None = None


def get_config() -> DengueCastConfig:
    """Get the default config singleton."""
    global _default_config
    if _default_config is None:
        _default_config = load_config()
    return _default_config
