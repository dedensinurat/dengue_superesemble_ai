"""Abstract ClimateProvider interface (MODEL_SPEC C.6)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class ClimateProvider(ABC):
    """Abstract base for all climate forecast providers.

    All implementations must respect the origin boundary:
    get() must NEVER return observed data for months > origin.
    """

    @abstractmethod
    def get(
        self,
        region_ids: list[str],
        origin: str,
        target_months: list[str],
    ) -> pd.DataFrame:
        """Return climate data for forecast months.

        Args:
            region_ids: List of region identifiers.
            origin: Latest month with observed case data (YYYY-MM).
            target_months: List of target months (all must be > origin).

        Returns:
            DataFrame with columns [region_id, target, member, tmean, tmin, tmax,
            dtr, precip, humidity, wind, nino34_anom, dmi_anom].
        """
        ...
