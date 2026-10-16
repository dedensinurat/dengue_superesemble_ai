"""Tests: package importable and config readable."""

from __future__ import annotations


def test_import_denguecast() -> None:
    """Package must be importable."""
    import denguecast

    assert denguecast.__version__ == "0.1.0"


def test_load_default_config() -> None:
    """Default config must load without errors."""
    from denguecast.config import load_config

    cfg = load_config()
    assert cfg.runtime.seed == 42
    assert cfg.horizon.max_lead == 6
    assert cfg.models.n_samples == 1000
    assert cfg.features.collinearity_threshold == 0.7


def test_config_no_magic_numbers() -> None:
    """Config must have all tunable parameters."""
    from denguecast.config import load_config

    cfg = load_config()
    # Verify key parameters are accessible
    assert cfg.models.top_k == 4
    assert cfg.models.max_candidates == 128
    assert cfg.evaluation.initial_train_years == 5
