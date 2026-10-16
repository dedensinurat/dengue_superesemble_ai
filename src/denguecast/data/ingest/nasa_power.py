"""NASA POWER API adapter for climate data (P3b, network-optional).

Queries the NASA POWER API for point/centroid coordinates.
Features:
- Mandatory disk caching (JSON / Parquet) with deterministic query hashing.
- Safe default: raises RuntimeError unless allow_network=True is explicitly set.
- Aggregates daily data to monthly climate_obs schema.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_CACHE_DIR = Path("data/interim/nasa_power_cache")


def get_query_hash(lat: float, lon: float, start: str, end: str) -> str:
    """Generate SHA-256 hash for query parameters."""
    key = f"{lat:.4f}_{lon:.4f}_{start}_{end}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def fetch_point_daily(
    lat: float,
    lon: float,
    start: str,
    end: str,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    allow_network: bool = False,
    mock_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch daily climate data from NASA POWER or cache.

    Args:
        lat: Latitude.
        lon: Longitude.
        start: Start date string (YYYYMMDD).
        end: End date string (YYYYMMDD).
        cache_dir: Directory for cached responses.
        allow_network: Must be True to invoke external API.
        mock_data: Optional mock data payload for testing.

    Returns:
        JSON response dict from NASA POWER API.
    """
    c_dir = Path(cache_dir)
    c_dir.mkdir(parents=True, exist_ok=True)
    q_hash = get_query_hash(lat, lon, start, end)
    cache_file = c_dir / f"power_{q_hash}.json"

    if cache_file.exists():
        return json.loads(cache_file.read_text(encoding="utf-8"))

    if mock_data is not None:
        cache_file.write_text(json.dumps(mock_data), encoding="utf-8")
        return mock_data

    if not allow_network:
        raise RuntimeError(
            f"Network access disabled and cache miss for ({lat}, {lon}, {start} to {end}). "
            "Use allow_network=True or --allow-network flag."
        )

    import httpx

    # NASA POWER Daily API endpoint
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params: dict[str, str | int | float] = {
        "parameters": "T2M,T2M_MIN,T2M_MAX,RH2M,PRECTOTCORR,WS2M",
        "community": "AG",
        "longitude": round(lon, 4),
        "latitude": round(lat, 4),
        "start": start,
        "end": end,
        "format": "JSON",
    }

    resp = httpx.get(url, params=params, timeout=30.0)
    resp.raise_for_status()
    data = resp.json()

    cache_file.write_text(json.dumps(data), encoding="utf-8")
    return data


def parse_power_daily_to_monthly(
    data: dict[str, Any],
    region_id: str,
) -> pd.DataFrame:
    """Parse NASA POWER JSON response into monthly climate_obs DataFrame."""
    params = data.get("properties", {}).get("parameter", {})
    if not params:
        return pd.DataFrame()

    dates = list(params.get("T2M", {}).keys())
    daily_rows = []

    for d_str in dates:
        # Date format YYYYMMDD
        date_fmt = f"{d_str[:4]}-{d_str[4:6]}-{d_str[6:8]}"
        m_str = f"{d_str[:4]}-{d_str[4:6]}"
        tmean = params.get("T2M", {}).get(d_str, np.nan)
        tmin = params.get("T2M_MIN", {}).get(d_str, np.nan)
        tmax = params.get("T2M_MAX", {}).get(d_str, np.nan)
        rh = params.get("RH2M", {}).get(d_str, np.nan)
        pr = params.get("PRECTOTCORR", {}).get(d_str, np.nan)
        ws = params.get("WS2M", {}).get(d_str, np.nan)

        # NASA POWER -999 indicates missing data
        tmean = np.nan if tmean == -999.0 else tmean
        tmin = np.nan if tmin == -999.0 else tmin
        tmax = np.nan if tmax == -999.0 else tmax
        rh = np.nan if rh == -999.0 else rh
        pr = np.nan if pr == -999.0 else pr
        ws = np.nan if ws == -999.0 else ws

        dtr = (tmax - tmin) if (pd.notna(tmax) and pd.notna(tmin)) else np.nan

        daily_rows.append(
            {
                "region_id": region_id,
                "date": date_fmt,
                "month": m_str,
                "tmean": tmean,
                "tmin": tmin,
                "tmax": tmax,
                "dtr": dtr,
                "humidity": rh,
                "precip": pr,
                "wind": ws,
            }
        )

    daily_df = pd.DataFrame(daily_rows)
    if daily_df.empty:
        return pd.DataFrame()

    # Aggregate to monthly
    monthly_rows = []
    for m, grp in daily_df.groupby("month"):
        monthly_rows.append(
            {
                "region_id": region_id,
                "month": m,
                "tmean": round(float(grp["tmean"].mean()), 2),
                "tmin": round(float(grp["tmin"].mean()), 2),
                "tmax": round(float(grp["tmax"].mean()), 2),
                "dtr": round(float(grp["dtr"].mean()), 2),
                "precip": round(float(grp["precip"].sum()), 1),
                "humidity": round(float(grp["humidity"].mean()), 1),
                "wind": round(float(grp["wind"].mean()), 2),
                "source": "nasa_power",
                "n_stations": 1,
                "valid_frac": 1.0,
                "quant_error": 0.0,
            }
        )

    return pd.DataFrame(monthly_rows)


def load(
    regions: pd.DataFrame,
    start: str,
    end: str,
    config: Any = None,
    allow_network: bool = False,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Load NASA POWER climate data for given regions and date range."""
    all_months = []
    for _, reg in regions.iterrows():
        rid = str(reg["region_id"])
        lat = float(reg["lat"])
        lon = float(reg["lon"])

        data = fetch_point_daily(
            lat=lat,
            lon=lon,
            start=start.replace("-", ""),
            end=end.replace("-", ""),
            cache_dir=cache_dir,
            allow_network=allow_network,
        )
        m_df = parse_power_daily_to_monthly(data, region_id=rid)
        all_months.append(m_df)

    if not all_months:
        return pd.DataFrame()
    return pd.concat(all_months, ignore_index=True)
