"""Recursive case lag reconstruction for h >= 2 (MODEL_SPEC C.5).

Model M0 (NB-GLM with region, month, population offset, and log(Ylag+1), without climate)
reconstructs the autoregressive lag series for multi-step ahead forecasts:
- At lead h=1: ylag is the observed cases at origin month.
- At lead h>=2: ylag is the recursively predicted cases from M0.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

logger = logging.getLogger(__name__)


class ReconstructLag:
    """Reconstructs lagged case values for multi-step ahead forecasting (C.5)."""

    def __init__(self) -> None:
        self._result: Any = None
        self._design_info: Any = None
        self._alpha: float = 0.25
        self._formula: str = ""
        self._latest_pop: dict[str, int] = {}
        self._cases_obs: dict[tuple[str, str], float] = {}
        self._region_groups: dict[str, str] = {}
        self._regional_means: dict[str, float] = {}
        self._overall_mean: float = 20.0

    def fit(self, train: pd.DataFrame) -> ReconstructLag:
        """Fit M0 model (region + month + log(Ylag+1), no climate).

        Args:
            train: Training DataFrame with cases, region_id, month, population, and ylag1/log_ylag1.

        Returns:
            self
        """
        train_df = train.copy()

        # 1. Prepare required columns
        if "month_of_year" not in train_df.columns and "month" in train_df.columns:
            train_df["month_of_year"] = train_df["month"].str.slice(5, 7).astype(int)

        if "log_pop_offset" not in train_df.columns and "population" in train_df.columns:
            train_df["log_pop_offset"] = np.log(train_df["population"] / 100_000.0)
        elif "log_pop_offset" not in train_df.columns:
            train_df["log_pop_offset"] = 0.0

        if "log_ylag1" not in train_df.columns and "ylag1" in train_df.columns:
            train_df["log_ylag1"] = np.log(np.maximum(0.0, train_df["ylag1"]) + 1.0)
        elif "log_ylag1" not in train_df.columns:
            train_df["log_ylag1"] = np.log(np.maximum(0.0, train_df["cases"].shift(1).fillna(20.0)) + 1.0)

        # Store lookups
        if "region_id" in train_df.columns and "month" in train_df.columns:
            for _, r in train_df.iterrows():
                self._cases_obs[(str(r["region_id"]), str(r["month"]))] = float(r["cases"])
                if "population" in r and not pd.isna(r["population"]):
                    self._latest_pop[str(r["region_id"])] = int(r["population"])
                if "region_group" in r and not pd.isna(r["region_group"]):
                    self._region_groups[str(r["region_id"])] = str(r["region_group"])

        if "region_id" in train_df.columns:
            self._regional_means = train_df.groupby("region_id")["cases"].mean().to_dict()

        self._overall_mean = float(train_df["cases"].mean()) if "cases" in train_df.columns else 20.0

        # 2. Seasonality term
        obs_per_reg = (
            train_df.groupby("region_id").size().min()
            if "region_id" in train_df.columns
            else len(train_df)
        )
        if obs_per_reg >= 72:
            season_term = "C(region_id):C(month_of_year) - 1"
        elif "region_group" in train_df.columns:
            season_term = "C(region_group):C(month_of_year) - 1"
        else:
            season_term = "C(region_id) + C(month_of_year)"

        formula = f"cases ~ {season_term} + log_ylag1"
        self._formula = formula

        # 3. Fit GLM NegativeBinomial
        try:
            y, X = patsy.dmatrices(formula, data=train_df, return_type="dataframe")
            self._design_info = X.design_info
            offset = train_df.loc[y.index, "log_pop_offset"].values

            # Estimate alpha via Poisson dispersion
            pois = sm.GLM(
                y,
                X,
                family=sm.families.Poisson(),
                offset=offset,
            ).fit(maxiter=20, tol=1e-4)
            mu_pois = pois.predict(X, offset=offset)
            res = y.values.squeeze() - mu_pois.values
            disp = float(np.mean((res**2 - mu_pois.values) / np.maximum(mu_pois.values**2, 1e-4)))
            self._alpha = float(np.clip(disp, 0.05, 2.5))

            # Fit GLM NB
            self._result = sm.GLM(
                y,
                X,
                family=sm.families.NegativeBinomial(alpha=self._alpha),
                offset=offset,
            ).fit(maxiter=25, tol=1e-4)
        except Exception as e:
            logger.warning(f"M0 fitting failed with {e}; will use fallback regional means.")
            self._result = None

        return self

    def reconstruct(
        self,
        origin: str,
        max_lead: int = 6,
        propagate_uncertainty: bool = False,
        seed: int = 42,
        cases_df: pd.DataFrame | None = None,
        pop_df: pd.DataFrame | None = None,
        regions_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Reconstruct ylag_hat for h=1..max_lead per region.

        Rules:
        - Lead h=1 uses actual observed cases at origin month.
        - Lead h>=2 uses the recursively predicted cases from M0.

        Args:
            origin: Origin month string 'YYYY-MM'.
            max_lead: Maximum forecast lead (default 6).
            propagate_uncertainty: If True, draws stochastic trajectories from NB.
            seed: Random seed for trajectory simulation.
            cases_df: Optional cases table to query observed cases at origin.
            pop_df: Optional population table.
            regions_df: Optional regions table.

        Returns:
            DataFrame with columns [region_id, origin, target, lead, ylag_hat].
        """
        rng = np.random.default_rng(seed)

        # Update observed lookup if cases_df provided
        if cases_df is not None:
            for _, r in cases_df.iterrows():
                self._cases_obs[(str(r["region_id"]), str(r["month"]))] = float(r["cases"])

        # Update population lookup if pop_df provided
        if pop_df is not None:
            for _, r in pop_df.iterrows():
                self._latest_pop[str(r["region_id"])] = int(r["population"])

        # Determine regions to evaluate
        if regions_df is not None and "region_id" in regions_df.columns:
            region_ids = [str(r) for r in regions_df["region_id"].unique()]
        elif self._latest_pop:
            region_ids = list(self._latest_pop.keys())
        elif self._regional_means:
            region_ids = list(self._regional_means.keys())
        else:
            region_ids = list({k[0] for k in self._cases_obs})

        origin_dt = pd.to_datetime(f"{origin}-01")
        sorted_rids = sorted(region_ids)

        # Initialize current lag for all regions at origin
        current_lags: dict[str, float] = {}
        for rid in sorted_rids:
            y_origin = self._cases_obs.get((rid, origin))
            if y_origin is None or np.isnan(y_origin):
                y_origin = self._regional_means.get(rid, self._overall_mean)
            current_lags[rid] = float(y_origin)

        records: list[dict[str, Any]] = []

        for h in range(1, max_lead + 1):
            target_dt = origin_dt + pd.DateOffset(months=h)
            target_str = target_dt.strftime("%Y-%m")
            target_cal_m = target_dt.month

            # 1. Record current lags for lead h
            for rid in sorted_rids:
                records.append(
                    {
                        "region_id": rid,
                        "origin": origin,
                        "target": target_str,
                        "lead": h,
                        "ylag_hat": round(current_lags[rid], 2),
                    }
                )

            # If this is the last lead, no need to project for h+1
            if h == max_lead:
                break

            # 2. Batch predict next step cases using M0
            if self._result is not None and self._design_info is not None:
                batch_rows = []
                offsets = []
                for rid in sorted_rids:
                    pop = self._latest_pop.get(rid, 1_000_000)
                    pop_off = float(np.log(pop / 100_000.0))
                    offsets.append(pop_off)
                    grp = self._region_groups.get(rid, "Java-Bali")
                    batch_rows.append(
                        {
                            "region_id": rid,
                            "region_group": grp,
                            "month_of_year": target_cal_m,
                            "log_ylag1": np.log(np.maximum(0.0, current_lags[rid]) + 1.0),
                            "log_pop_offset": pop_off,
                        }
                    )

                batch_df = pd.DataFrame(batch_rows)
                try:
                    dmat = patsy.build_design_matrices([self._design_info], batch_df)[0]
                    mu_vec = self._result.predict(dmat, offset=np.array(offsets))
                    for i, rid in enumerate(sorted_rids):
                        mu = max(0.1, float(mu_vec[i]))
                        if propagate_uncertainty:
                            p = 1.0 / (1.0 + self._alpha * mu)
                            n = 1.0 / self._alpha
                            current_lags[rid] = max(0.0, float(rng.negative_binomial(n, p)))
                        else:
                            current_lags[rid] = mu
                except Exception as e:
                    logger.debug(f"Batch M0 prediction failed: {e}")
                    for rid in sorted_rids:
                        current_lags[rid] = self._regional_means.get(rid, self._overall_mean)
            else:
                for rid in sorted_rids:
                    current_lags[rid] = self._regional_means.get(rid, self._overall_mean)

        return pd.DataFrame(records)
