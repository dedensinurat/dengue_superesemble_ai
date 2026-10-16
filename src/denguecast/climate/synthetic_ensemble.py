"""SyntheticEnsembleProvider: climatology + correlated noise, 20 members."""

from __future__ import annotations

import numpy as np
import pandas as pd

from denguecast.climate.provider import ClimateProvider


class SyntheticEnsembleProvider(ClimateProvider):
    """Climatology with correlated noise for testing ensemble pathways."""

    def __init__(
        self,
        climatology_provider: ClimateProvider,
        n_members: int = 20,
        seed: int = 42,
    ) -> None:
        self._clim = climatology_provider
        self._n_members = n_members
        self._seed = seed

    def get(
        self,
        region_ids: list[str],
        origin: str,
        target_months: list[str],
    ) -> pd.DataFrame:
        """Return ensemble of climate forecasts."""
        base_df = self._clim.get(region_ids, origin, target_months)
        rng = np.random.default_rng(self._seed)

        all_members = []
        n_rows = len(base_df)

        for m in range(self._n_members):
            member_df = base_df.copy()
            member_df["member"] = m

            if m > 0:
                # Add correlated noise to each variable
                t_noise = rng.normal(0, 0.4, n_rows)
                p_factor = np.clip(1.0 + rng.normal(0, 0.12, n_rows), 0.2, 2.5)
                h_noise = rng.normal(0, 2.5, n_rows)
                dtr_noise = rng.normal(0, 0.3, n_rows)
                wind_noise = rng.normal(0, 0.4, n_rows)
                nino_noise = rng.normal(0, 0.15, n_rows)

                member_df["tmean"] = (member_df["tmean"] + t_noise).round(2)
                member_df["dtr"] = np.clip(member_df["dtr"] + dtr_noise, 0.5, 30.0).round(2)
                member_df["tmin"] = (member_df["tmean"] - member_df["dtr"] / 2.0).round(2)
                member_df["tmax"] = (member_df["tmean"] + member_df["dtr"] / 2.0).round(2)
                member_df["precip"] = np.clip(member_df["precip"] * p_factor, 0.0, 2000.0).round(1)
                member_df["humidity"] = np.clip(member_df["humidity"] + h_noise, 10.0, 100.0).round(
                    1
                )
                member_df["wind"] = np.clip(member_df["wind"] + wind_noise, 0.0, 50.0).round(2)
                member_df["nino34_anom"] = (member_df["nino34_anom"] + nino_noise).round(3)

            all_members.append(member_df)

        return pd.concat(all_members, ignore_index=True)
