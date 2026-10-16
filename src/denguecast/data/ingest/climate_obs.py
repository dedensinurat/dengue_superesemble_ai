"""Generic climate observation adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from denguecast.data.schema import validate_climate_obs


def load(source: str | Path, config: Any = None) -> pd.DataFrame:
    """Load climate observations from CSV.

    Args:
        source: File path to CSV.
        config: Optional configuration.

    Returns:
        Validated climate_obs DataFrame.
    """
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"Climate observation file not found: {path}")

    df = pd.read_csv(path)

    # Calculate DTR if tmax and tmin exist and dtr is missing
    if "dtr" not in df.columns and "tmax" in df.columns and "tmin" in df.columns:
        df["dtr"] = (df["tmax"] - df["tmin"]).round(2)

    validate_climate_obs(df)
    return df
