"""ClimatologyProvider: calendar-month climatology (MODEL_SPEC C.6, MVP)."""

from __future__ import annotations

import pandas as pd

from denguecast.climate.provider import ClimateProvider


class ClimatologyProvider(ClimateProvider):
    """Uses up to 10-year calendar mean before origin as the single ensemble member."""

    def __init__(
        self,
        climate_obs: pd.DataFrame,
        enso_obs: pd.DataFrame | None = None,
        window_years: int = 10,
    ) -> None:
        """Initialize with historical observations.

        Args:
            climate_obs: Observed climate data up to any date.
            enso_obs: Optional ENSO/index observations.
            window_years: Number of historical years before origin to average.
        """
        self._climate_obs = climate_obs.copy()
        self._enso_obs = enso_obs.copy() if enso_obs is not None else None
        self._window_years = window_years

    def get(
        self,
        region_ids: list[str],
        origin: str,
        target_months: list[str],
    ) -> pd.DataFrame:
        """Return climatology forecasts for target months.

        STRICT RULE: Only data with month <= origin is used.
        Target months must all be > origin.

        Args:
            region_ids: List of region IDs.
            origin: Origin month string (YYYY-MM).
            target_months: Target month strings (YYYY-MM).

        Returns:
            DataFrame [region_id, target, member, tmean, tmin, tmax, dtr,
                       precip, humidity, wind, nino34_anom, dmi_anom].
        """
        # Strict anti-leakage filter
        hist_climate = self._climate_obs[self._climate_obs["month"] <= origin].copy()
        if hist_climate.empty:
            raise ValueError(f"No climate data available on or before origin '{origin}'")

        # Restrict to window_years before origin
        origin_year = int(origin.split("-")[0])
        min_year = origin_year - self._window_years
        hist_climate["year"] = hist_climate["month"].str.slice(0, 4).astype(int)
        hist_climate = hist_climate[hist_climate["year"] >= min_year]
        hist_climate["cal_month"] = hist_climate["month"].str.slice(5, 7).astype(int)

        clim_vars = ["tmean", "tmin", "tmax", "dtr", "precip", "humidity", "wind"]
        # Filter variables present
        avail_vars = [c for c in clim_vars if c in hist_climate.columns]

        # Calculate climatology: mean by (region_id, cal_month)
        grouped = hist_climate.groupby(["region_id", "cal_month"])[avail_vars].mean()
        # Fallback mean across all regions for each cal_month if a specific region has no data
        month_fallback = hist_climate.groupby("cal_month")[avail_vars].mean()

        # ENSO persistence at origin (C.13: persistence_forecast)
        nino_at_origin = 0.0
        dmi_at_origin = 0.0
        if self._enso_obs is not None:
            hist_enso = self._enso_obs[self._enso_obs["month"] <= origin]
            if not hist_enso.empty:
                latest_enso = hist_enso.sort_values("month").iloc[-1]
                nino_at_origin = float(latest_enso.get("nino34_anom", latest_enso.get("oni", 0.0)))
                dmi_at_origin = float(latest_enso.get("dmi", latest_enso.get("dmi_anom", 0.0)))

        rows = []
        for target in target_months:
            cal_m = int(target.split("-")[1])
            for rid in region_ids:
                row: dict[str, object] = {
                    "region_id": rid,
                    "target": target,
                    "member": 0,
                    "nino34_anom": nino_at_origin,
                    "dmi_anom": dmi_at_origin,
                }

                # Fetch climatological mean
                if (rid, cal_m) in grouped.index:
                    vals = grouped.loc[(rid, cal_m)]
                elif cal_m in month_fallback.index:
                    vals = month_fallback.loc[cal_m]
                else:
                    vals = pd.Series({v: 0.0 for v in avail_vars})

                for v in clim_vars:
                    row[v] = float(vals[v]) if v in vals and pd.notna(vals[v]) else 0.0

                rows.append(row)

        cols = [
            "region_id",
            "target",
            "member",
            "tmean",
            "tmin",
            "tmax",
            "dtr",
            "precip",
            "humidity",
            "wind",
            "nino34_anom",
            "dmi_anom",
        ]
        return pd.DataFrame(rows, columns=cols)
