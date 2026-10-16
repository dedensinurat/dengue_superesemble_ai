"""Ingest adapters for DengueCast data sources."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from denguecast.data.ingest import bmkg_dataonline as bmkg_adapter
from denguecast.data.ingest import cases as cases_adapter
from denguecast.data.ingest import climate_obs as climate_adapter
from denguecast.data.ingest import opendengue as opendengue_adapter

CASES_ADAPTERS: dict[str, Callable[..., Any]] = {
    "csv": cases_adapter.load,
    "opendengue": opendengue_adapter.load,
}

CLIMATE_ADAPTERS: dict[str, Callable[..., Any]] = {
    "csv": climate_adapter.load,
    "bmkg_station": bmkg_adapter.load_station_file,
}


def register_cases_adapter(name: str, adapter_fn: Callable[..., Any]) -> None:
    """Register a new cases adapter."""
    CASES_ADAPTERS[name.lower()] = adapter_fn


def register_climate_adapter(name: str, adapter_fn: Callable[..., Any]) -> None:
    """Register a new climate adapter."""
    CLIMATE_ADAPTERS[name.lower()] = adapter_fn


def get_cases_adapter(name: str) -> Callable[..., Any]:
    """Retrieve cases adapter by name."""
    clean = name.lower()
    if clean not in CASES_ADAPTERS:
        raise ValueError(
            f"Unknown cases adapter '{name}'. Available: {list(CASES_ADAPTERS.keys())}"
        )
    return CASES_ADAPTERS[clean]


def get_climate_adapter(name: str) -> Callable[..., Any]:
    """Retrieve climate adapter by name."""
    clean = name.lower()
    if clean not in CLIMATE_ADAPTERS:
        raise ValueError(
            f"Unknown climate adapter '{name}'. Available: {list(CLIMATE_ADAPTERS.keys())}"
        )
    return CLIMATE_ADAPTERS[clean]
