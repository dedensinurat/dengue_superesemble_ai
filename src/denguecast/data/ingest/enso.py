"""Adapter for ENSO (Nino 3.4) and IOD (DMI) data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from denguecast.data.schema import validate_enso


def load_nino34(path: str | Path) -> pd.DataFrame:
    """Load Nino 3.4 anomaly data from CSV or NOAA CPC matrix text file.

    Returns:
        DataFrame with columns [month, nino34_anom].
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"ENSO file not found: {p}")

    # Check if CSV with columns [month, nino34_anom]
    try:
        df = pd.read_csv(p)
        if "month" in df.columns and ("nino34_anom" in df.columns or "oni" in df.columns):
            if "nino34_anom" not in df.columns and "oni" in df.columns:
                df["nino34_anom"] = df["oni"]
            res = df[["month", "nino34_anom"]].dropna().copy()
            res["month"] = res["month"].astype(str)
            res["nino34_anom"] = res["nino34_anom"].astype(float)
            validate_enso(res)
            return res
    except Exception:
        pass

    # CPC format: YEAR DJF JFM ... or YEAR JAN FEB ...
    lines = p.read_text(encoding="utf-8").strip().splitlines()
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 13 and parts[0].isdigit() and len(parts[0]) == 4:
            year = parts[0]
            for m_idx in range(1, 13):
                try:
                    val = float(parts[m_idx])
                    if val < -90.0:  # Missing code like -99.9
                        continue
                    month_str = f"{year}-{m_idx:02d}"
                    rows.append({"month": month_str, "nino34_anom": val})
                except (ValueError, IndexError):
                    continue

    result = pd.DataFrame(rows)
    validate_enso(result)
    return result


def load(path: str | Path, config: Any = None) -> pd.DataFrame:
    """Generic load entry point."""
    return load_nino34(path)
