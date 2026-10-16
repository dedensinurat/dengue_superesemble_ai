"""SeasonalForecastProvider: real seasonal forecast files (Tahap B, stub)."""

from __future__ import annotations

import pandas as pd

from denguecast.climate.provider import ClimateProvider


class SeasonalForecastProvider(ClimateProvider):
    """Reads NetCDF seasonal forecast files (e.g., Copernicus C3S).

    STUB: Implement in Tahap B.
    Requires hindcast data for expanding-window evaluation.
    """

    def get(self, region_ids: list[str], origin: str, target_months: list[str]) -> pd.DataFrame:
        raise NotImplementedError("Tahap B: implement SeasonalForecastProvider")
