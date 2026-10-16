"""Baseline models: B1 SeasonalExpandingMean, B2 SeasonalNB (MODEL_SPEC C.11)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import glm

from denguecast.models.base import ForecastModel, ModelSpec


class SeasonalExpandingMean(ForecastModel):
    """B1: cumulative mean per region-month from training data."""

    def __init__(self) -> None:
        self._means: dict[tuple[str, int], float] = {}
        self._month_means: dict[int, float] = {}
        self._dispersions: dict[str, float] = {}
        self._overall_alpha: float = 0.25

    def fit(self, train: pd.DataFrame) -> SeasonalExpandingMean:
        train_df = train.copy()
        if "month_of_year" not in train_df.columns:
            train_df["month_of_year"] = train_df["month"].str.slice(5, 7).astype(int)

        # Calculate mean by (region_id, month_of_year)
        grouped = train_df.groupby(["region_id", "month_of_year"])["cases"].mean()
        self._means = grouped.to_dict()

        # Fallback monthly mean
        self._month_means = train_df.groupby("month_of_year")["cases"].mean().to_dict()

        # Dispersion alpha per region: var = mu + alpha * mu^2
        for rid, grp in train_df.groupby("region_id"):
            m = float(grp["cases"].mean())
            v = float(grp["cases"].var())
            if m > 0 and v > m:
                alpha = (v - m) / (m**2)
                self._dispersions[str(rid)] = float(np.clip(alpha, 0.05, 2.0))
            else:
                self._dispersions[str(rid)] = 0.2

        overall_m = float(train_df["cases"].mean())
        overall_v = float(train_df["cases"].var())
        if overall_m > 0 and overall_v > overall_m:
            self._overall_alpha = float(np.clip((overall_v - overall_m) / (overall_m**2), 0.05, 2.0))

        return self

    def predict_samples(self, X: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_rows = len(X)
        samples = np.zeros((n_rows, n_samples), dtype=int)

        x_df = X.copy()
        if "month_of_year" not in x_df.columns and "month" in x_df.columns:
            x_df["month_of_year"] = x_df["month"].str.slice(5, 7).astype(int)
        elif "month_of_year" not in x_df.columns and "target" in x_df.columns:
            x_df["month_of_year"] = x_df["target"].str.slice(5, 7).astype(int)

        for i, (_, row) in enumerate(x_df.iterrows()):
            rid = str(row["region_id"])
            moy = int(row.get("month_of_year", 1))

            mu = self._means.get((rid, moy), self._month_means.get(moy, 20.0))
            mu = max(0.1, float(mu))
            alpha = self._dispersions.get(rid, self._overall_alpha)

            # Draw from Negative Binomial: mean mu, var mu + alpha*mu^2
            p = 1.0 / (1.0 + alpha * mu)
            n = 1.0 / alpha
            draws = rng.negative_binomial(n, p, size=n_samples)
            samples[i, :] = np.maximum(0, draws)

        return samples

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(model_id="B1_seasonal_mean", features=[], description="B1 Cumulative Seasonal Mean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.spec.model_id,
            "type": "SeasonalExpandingMean",
            "n_region_months": len(self._means),
        }


class SeasonalNB(ForecastModel):
    """B2: NB-GLM with region + month effects, population offset; no climate, no ylag."""

    def __init__(self) -> None:
        self._result: Any = None
        self._alpha: float = 0.25
        self._fallback_means: dict[tuple[str, int], float] = {}

    def fit(self, train: pd.DataFrame) -> SeasonalNB:
        train_df = train.copy()
        if "month_of_year" not in train_df.columns:
            train_df["month_of_year"] = train_df["month"].str.slice(5, 7).astype(int)
        if "log_pop_offset" not in train_df.columns:
            train_df["log_pop_offset"] = np.log(train_df["population"] / 100_000.0)

        # Precompute fallback means
        self._fallback_means = train_df.groupby(["region_id", "month_of_year"])["cases"].mean().to_dict()

        # Seasonality structure (P5b Rule 3: same structure as NBGLM)
        obs_per_reg = (
            train_df.groupby("region_id").size().min()
            if "region_id" in train_df.columns
            else len(train_df)
        )
        if obs_per_reg >= 72:
            formula = "cases ~ C(region_id):C(month_of_year) - 1"
        elif "region_group" in train_df.columns:
            formula = "cases ~ C(region_group):C(month_of_year) - 1"
        else:
            formula = "cases ~ C(region_id) + C(month_of_year)"

        # Estimate alpha using Poisson deviance dispersion
        try:
            pois = glm(
                formula,
                data=train_df,
                family=sm.families.Poisson(),
                offset=train_df["log_pop_offset"],
            ).fit(maxiter=20, tol=1e-4)
            pearson_disp = float(pois.pearson_chi2 / pois.df_resid)
            self._alpha = float(
                np.clip((pearson_disp - 1.0) / np.maximum(train_df["cases"].mean(), 1.0), 0.05, 1.5)
            )
            # Fit NegativeBinomial GLM
            self._result = glm(
                formula,
                data=train_df,
                family=sm.families.NegativeBinomial(alpha=self._alpha),
                offset=train_df["log_pop_offset"],
            ).fit(maxiter=25, tol=1e-4)
        except Exception:
            # Fallback to simple Poisson
            self._result = glm("cases ~ C(month_of_year)", data=train_df, family=sm.families.Poisson()).fit()

        return self

    def predict_samples(self, X: pd.DataFrame, n_samples: int = 1000, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        n_rows = len(X)
        samples = np.zeros((n_rows, n_samples), dtype=int)

        x_df = X.copy()
        if "month_of_year" not in x_df.columns and "target" in x_df.columns:
            x_df["month_of_year"] = x_df["target"].str.slice(5, 7).astype(int)
        if "log_pop_offset" not in x_df.columns and "population" in x_df.columns:
            x_df["log_pop_offset"] = np.log(x_df["population"] / 100_000.0)
        elif "log_pop_offset" not in x_df.columns:
            x_df["log_pop_offset"] = 0.0

        try:
            pred_mu = self._result.predict(x_df, offset=x_df["log_pop_offset"])
        except Exception:
            # Fallback to empirical means
            pred_mu = pd.Series([
                self._fallback_means.get((str(r["region_id"]), int(r.get("month_of_year", 1))), 20.0)
                for _, r in x_df.iterrows()
            ])

        for i, mu_val in enumerate(pred_mu):
            mu = max(0.1, float(mu_val))
            p = 1.0 / (1.0 + self._alpha * mu)
            n = 1.0 / self._alpha
            draws = rng.negative_binomial(n, p, size=n_samples)
            samples[i, :] = np.maximum(0, draws)

        return samples

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(model_id="B2_seasonal_nb", features=[], description="B2 Seasonal NB-GLM")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.spec.model_id,
            "type": "SeasonalNB",
            "alpha": self._alpha,
        }
