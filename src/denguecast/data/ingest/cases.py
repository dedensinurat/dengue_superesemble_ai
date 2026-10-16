"""Generic CSV adapter for dengue case data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from denguecast.data.schema import validate_cases


def load(source: str | Path, config: Any = None) -> pd.DataFrame:
    """Load case data from CSV source.

    Args:
        source: Filepath or source identifier.
        config: Optional config object with column mapping.

    Returns:
        Validated cases DataFrame.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Cases source file not found: {path}")

    df = pd.read_csv(path)

    # Standardize column names if column_map provided in config
    if config is not None and hasattr(config, "cases") and hasattr(config.cases, "column_map"):
        df = df.rename(columns=config.cases.column_map)

    # Ensure required columns exist
    if "definition" not in df.columns:
        df["definition"] = "clinical"

    df["cases"] = df["cases"].astype(int)
    df["month"] = df["month"].astype(str)
    df["region_id"] = df["region_id"].astype(str)

    validate_cases(df)
    return df
