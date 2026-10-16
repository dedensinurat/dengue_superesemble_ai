"""Land cover data adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from denguecast.data.schema import validate_landcover


def load(path: str | Path, config: Any = None) -> pd.DataFrame:
    """Load land cover data from CSV.

    Returns:
        DataFrame with columns [region_id, year, urban_pct, periurban_pct].
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Landcover file not found: {p}")

    df = pd.read_csv(p)
    df["region_id"] = df["region_id"].astype(str)
    df["year"] = df["year"].astype(int)
    df["urban_pct"] = df["urban_pct"].astype(float)
    df["periurban_pct"] = df["periurban_pct"].astype(float)

    validate_landcover(df)
    return df
