"""NB-GLM candidate model (MODEL_SPEC C.3 and C.13).

Negative Binomial GLM with climate features, autoregressive lagged cases,
and regional seasonality.

Uncertainty quantification:
- Sample beta ~ MVN(beta_hat, Cov + 1e-6*I)
- Sample Y ~ NB(mu(beta), alpha)
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

from denguecast.models.base import ForecastModel, ModelSpec

logger = logging.getLogger(__name__)


class NBGLM(ForecastModel):
    """Negative Binomial GLM with climate features and lagged cases.

    Supports:
    - Region x month seasonal interactions (or group x month fallback when obs < 72).
    - Climate by region group interactions with small-group pooling (P5b).
    - Method-of-moments dispersion estimation with fallback from discrete MLE.
    - Vectorized MVN parameter sampling for probabilistic forecasts.
    """

    def __init__(
        self,
        features: list[str] | None = None,
        climate_by_group: bool = False,
        model_id: str | None = None,
    ) -> None:
        self._features = list(features) if features is not None else []
        self._climate_by_group = climate_by_group
        self._model_id = model_id or (
            f"nbglm_{'_'.join(self._features[:3])}" if self._features else "nbglm_base"
        )
        self._result: Any = None
        self._design_info: Any = None
        self._formula: str = ""
        self._alpha: float = 0.25
        self._params: pd.Series | None = None
        self._cov: np.ndarray | None = None
        self._converged: bool = False
        self._conv_warning: str | None = None
        self._season_structure: str = "common_month"
        self._n_obs: int = 0
        self._fit_date: str = ""
        self._clim_group_map: dict[str, str] = {}

    def fit(self, train: pd.DataFrame) -> NBGLM:
        """Fit Negative Binomial GLM on training data.

        Args:
            train: Training DataFrame with cases, region_id, month (or month_of_year),
                   population (or log_pop_offset), log_ylag1, and predictor features.

        Returns:
            self
        """
        train_df = train.copy()
        self._n_obs = len(train_df)
        self._fit_date = datetime.now(UTC).isoformat()

        # 1. Prepare required columns
        if "month_of_year" not in train_df.columns and "month" in train_df.columns:
            train_df["month_of_year"] = train_df["month"].str.slice(5, 7).astype(int)
        if "log_pop_offset" not in train_df.columns and "population" in train_df.columns:
            train_df["log_pop_offset"] = np.log(train_df["population"] / 100_000.0)
        elif "log_pop_offset" not in train_df.columns:
            train_df["log_pop_offset"] = 0.0

        if "log_ylag1" not in train_df.columns and "ylag1" in train_df.columns:
            train_df["log_ylag1"] = np.log(np.maximum(0.0, train_df["ylag1"]) + 1.0)

        # 2. Seasonality structure (P5b Rule 1)
        # Check minimum observations per region
        obs_per_region = (
            train_df.groupby("region_id").size().min()
            if "region_id" in train_df.columns
            else len(train_df)
        )

        has_region_group = "region_group" in train_df.columns
        if obs_per_region >= 72:
            season_term = "C(region_id):C(month_of_year) - 1"
            self._season_structure = "region_x_month"
        elif has_region_group:
            logger.debug(
                f"Min observations per region ({obs_per_region}) < 72. "
                "Using group x month seasonal structure."
            )
            season_term = "C(region_group):C(month_of_year) - 1"
            self._season_structure = "group_x_month"
        else:
            logger.debug(
                f"Min observations per region ({obs_per_region}) < 72 and no region_group found. "
                "Using common month seasonal structure."
            )
            season_term = "C(region_id) + C(month_of_year)"
            self._season_structure = "common_month"

        # 3. Autoregressive and climate predictor terms
        predictor_terms: list[str] = []

        # Ensure autoregressive log_ylag1 is included if present and not already in features
        if "log_ylag1" in train_df.columns and "log_ylag1" not in self._features:
            predictor_terms.append("log_ylag1")

        # Handle climate_by_group (P5b Rule 2)
        if self._climate_by_group and has_region_group:
            # Check groups with < 3 provinces (Maluku, Papua, Nusa Tenggara)
            n_prov_per_group = train_df.groupby("region_group")["region_id"].nunique()
            self._clim_group_map = {
                str(grp): (str(grp) if n_prov_per_group.get(grp, 0) >= 3 else "Small_Groups")
                for grp in n_prov_per_group.index
            }
            train_df["clim_group"] = train_df["region_group"].astype(str).map(self._clim_group_map)

            for feat in self._features:
                if feat not in train_df.columns:
                    continue
                # Interact precip and humidity with clim_group
                feat_lower = feat.lower()
                if "precip" in feat_lower or "humidity" in feat_lower:
                    predictor_terms.append(f"{feat}:C(clim_group)")
                else:
                    predictor_terms.append(feat)
        else:
            for feat in self._features:
                if feat in train_df.columns:
                    predictor_terms.append(feat)

        # 4. Construct patsy formula
        if predictor_terms:
            formula = f"cases ~ {season_term} + " + " + ".join(predictor_terms)
        else:
            formula = f"cases ~ {season_term}"

        self._formula = formula
        logger.debug(f"Fitting NBGLM with formula: {formula}")

        # 5. Build design matrices
        y, X = patsy.dmatrices(formula, data=train_df, return_type="dataframe")
        self._design_info = X.design_info
        offset = train_df.loc[y.index, "log_pop_offset"].values

        # 6. Estimate alpha dispersion via method-of-moments on Poisson fit
        try:
            pois = sm.GLM(
                y,
                X,
                family=sm.families.Poisson(),
                offset=offset,
            ).fit(maxiter=25, tol=1e-4)
            mu_pois = pois.predict(X, offset=offset)
            res_pois = y.values.squeeze() - mu_pois.values
            disp = float(np.mean((res_pois**2 - mu_pois.values) / np.maximum(mu_pois.values**2, 1e-4)))
            moment_alpha = float(np.clip(disp, 0.05, 2.5))
        except Exception as e:
            logger.warning(f"Poisson alpha estimation encountered issue: {e}. Defaulting alpha=0.25.")
            moment_alpha = 0.25

        self._alpha = moment_alpha

        # 7. Fit Negative Binomial model
        # Try discrete MLE first if parameter space is small (<= 40), otherwise use IRLS GLM directly
        fit_success = False
        if X.shape[1] <= 40:
            try:
                discrete_nb = sm.NegativeBinomial(
                    y, X, offset=offset, loglike_method="nb2"
                )
                res = discrete_nb.fit(disp=False, maxiter=50)
                if getattr(res, "mle_retvals", {}).get("converged", False):
                    self._result = res
                    self._alpha = float(np.clip(res.params.get("alpha", moment_alpha), 0.02, 3.0))
                    self._params = res.params.drop("alpha", errors="ignore")
                    self._cov = res.cov_params().drop("alpha", errors="ignore").drop("alpha", axis=1, errors="ignore").values
                    self._converged = True
                    fit_success = True
            except Exception as e:
                logger.info(f"Discrete NegativeBinomial fit fallback: {e}")

        if not fit_success:
            # Fallback GLM NegativeBinomial with method of moments alpha
            glm_nb = sm.GLM(
                y,
                X,
                family=sm.families.NegativeBinomial(alpha=self._alpha),
                offset=offset,
            )
            res = glm_nb.fit(maxiter=25, tol=1e-4)
            self._result = res
            self._params = res.params
            self._cov = res.cov_params().values
            self._converged = bool(res.converged)
            if not self._converged:
                self._conv_warning = "GLM IRLS did not fully converge within max iterations."

        return self

    def predict_samples(
        self,
        X: pd.DataFrame,
        n_samples: int = 1000,
        seed: int = 42,
    ) -> np.ndarray:
        """Sample beta ~ MVN(beta_hat, Cov), then Y ~ NB(mu(beta), alpha).

        Args:
            X: Feature DataFrame with candidate predictors.
            n_samples: Number of predictive draws.
            seed: Random seed for reproducibility.

        Returns:
            Array of shape [n_rows, n_samples] with non-negative integer counts.
        """
        if self._result is None or self._params is None or self._cov is None:
            raise RuntimeError("Model must be fit before calling predict_samples().")

        x_df = X.copy()
        n_rows = len(x_df)

        # 1. Fill default columns
        if "month_of_year" not in x_df.columns:
            if "month" in x_df.columns:
                x_df["month_of_year"] = x_df["month"].str.slice(5, 7).astype(int)
            elif "target" in x_df.columns:
                x_df["month_of_year"] = x_df["target"].str.slice(5, 7).astype(int)
            else:
                x_df["month_of_year"] = 1

        if "log_pop_offset" not in x_df.columns:
            if "population" in x_df.columns:
                x_df["log_pop_offset"] = np.log(x_df["population"] / 100_000.0)
            else:
                x_df["log_pop_offset"] = 0.0

        if "log_ylag1" not in x_df.columns:
            if "ylag1" in x_df.columns:
                x_df["log_ylag1"] = np.log(np.maximum(0.0, x_df["ylag1"]) + 1.0)
            else:
                x_df["log_ylag1"] = np.log(21.0)

        if self._climate_by_group and "region_group" in x_df.columns:
            x_df["clim_group"] = (
                x_df["region_group"].astype(str).map(self._clim_group_map).fillna("Small_Groups")
            )

        # Fill NaNs in predictor features
        for f in self._features:
            if f in x_df.columns:
                x_df[f] = x_df[f].fillna(0.0)
            else:
                x_df[f] = 0.0

        # 2. Build test design matrix
        try:
            design_matrices = patsy.build_design_matrices([self._design_info], x_df)
            X_mat = np.asarray(design_matrices[0])
            if X_mat.shape[0] != n_rows:
                logger.warning(f"Design matrix rows ({X_mat.shape[0]}) mismatch n_rows ({n_rows}). Using fallback.")
                return np.zeros((n_rows, n_samples), dtype=int)
        except Exception as e:
            logger.warning(f"Error building design matrix for prediction: {e}. Using fallback.")
            return np.zeros((n_rows, n_samples), dtype=int)

        # 3. Draw parameter samples from MVN with diagonal regularization
        k = len(self._params)
        cov_sym = 0.5 * (self._cov + self._cov.T) + 1e-6 * np.eye(k)
        # Ensure positive semi-definiteness via eigenvalue clipping
        eigvals, eigvecs = np.linalg.eigh(cov_sym)
        eigvals = np.maximum(eigvals, 1e-7)
        cov_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        rng = np.random.default_rng(seed)
        beta_samples = rng.multivariate_normal(
            mean=self._params.values, cov=cov_psd, size=n_samples
        )  # shape: (n_samples, k)

        # 4. Compute linear predictor: eta = X @ beta^T + offset
        eta = X_mat @ beta_samples.T  # shape: (n_rows, n_samples)
        offsets = x_df["log_pop_offset"].values[:, None]
        eta = eta + offsets
        mu = np.exp(np.clip(eta, -20.0, 20.0))

        # 5. Draw counts from Negative Binomial
        # Var = mu + alpha * mu^2 => p = 1 / (1 + alpha * mu), n = 1 / alpha
        p = 1.0 / (1.0 + self._alpha * mu)
        n = 1.0 / self._alpha
        draws = rng.negative_binomial(n, p)

        return np.maximum(0, draws).astype(int)

    def summary_frame(self) -> pd.DataFrame:
        """Return coefficients, SE, p-values, and 95% confidence intervals."""
        if self._result is None or self._params is None or self._cov is None:
            raise RuntimeError("Model must be fit before calling summary_frame().")

        params = self._params
        bse = pd.Series(np.sqrt(np.diag(self._cov)), index=params.index)
        z_vals = params / np.maximum(bse, 1e-9)

        # 95% CI
        ci_lower = params - 1.96 * bse
        ci_upper = params + 1.96 * bse

        # p-values via normal approx
        from scipy import stats

        p_vals = 2 * (1 - stats.norm.cdf(np.abs(z_vals)))

        return pd.DataFrame(
            {
                "term": params.index,
                "coef": params.values.round(4),
                "std_err": bse.values.round(4),
                "z_value": z_vals.values.round(3),
                "p_value": p_vals.round(4),
                "ci_lower": ci_lower.values.round(4),
                "ci_upper": ci_upper.values.round(4),
            }
        )

    def model_card(self) -> dict[str, Any]:
        """Return structured model metadata card (P5b Rule 4)."""
        return {
            "model_id": self.spec.model_id,
            "features": self._features,
            "climate_by_group": self._climate_by_group,
            "season_structure": self._season_structure,
            "n_obs": self._n_obs,
            "fit_date": self._fit_date,
            "alpha": round(float(self._alpha), 4),
            "converged": self._converged,
            "convergence_warning": self._conv_warning,
            "formula": self._formula,
        }

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec(
            model_id=self._model_id,
            features=self._features,
            climate_by_group=self._climate_by_group,
            description=f"NBGLM with {len(self._features)} features (climate_by_group={self._climate_by_group})",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.spec.model_id,
            "type": "NBGLM",
            "features": self._features,
            "climate_by_group": self._climate_by_group,
            "alpha": float(self._alpha),
            "n_params": len(self._params) if self._params is not None else 0,
            "converged": self._converged,
        }
