"""Shared fixtures for DengueCast tests."""

from __future__ import annotations

import pytest

from denguecast.config import DengueCastConfig


@pytest.fixture
def default_config() -> DengueCastConfig:
    from denguecast.config import load_config

    return load_config()
