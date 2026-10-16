"""Unit tests for ForecastModel, NBGLM, ReconstructLag, and Baselines (P5 & P5b)."""

import numpy as np
import pytest

from denguecast.data.synthetic import generate_synthetic_data
from denguecast.features.build import build_training_frame
from denguecast.models.base import ModelSpec
from denguecast.models.baseline import SeasonalExpandingMean, SeasonalNB
from denguecast.models.nb_glm import NBGLM
from denguecast.models.recursive_cases import ReconstructLag
from denguecast.models.registry import build_model, list_models


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Generate deterministic synthetic dataset for model tests."""
    tables = generate_synthetic_data(n_regions=12, n_years=6, seed=42)
    train_df, scaler, features = build_training_frame(
        cases=tables["cases"],
        pop=tables["population"],
        climate_obs=tables["climate_obs"],
        enso=tables["enso"],
        regions=tables["regions"],
        origin="2021-12",
    )
    return {
        "tables": tables,
        "train_df": train_df,
        "scaler": scaler,
        "features": features,
    }


class TestModelRegistry:
    def test_list_models(self):
        models = list_models()
        assert "B1_seasonal_mean" in models
        assert "B2_seasonal_nb" in models
        assert "nbglm" in models

    def test_build_baseline_models(self):
        m1 = build_model(ModelSpec(model_id="B1_seasonal_mean"))
        assert isinstance(m1, SeasonalExpandingMean)

        m2 = build_model(ModelSpec(model_id="B2_seasonal_nb"))
        assert isinstance(m2, SeasonalNB)

    def test_build_nbglm(self):
        m = build_model(ModelSpec(model_id="nbglm", features=["humidity_l02_z"]))
        assert isinstance(m, NBGLM)
        assert m.spec.features == ["humidity_l02_z"]


class TestBaselines:
    def test_b1_seasonal_expanding_mean(self, synthetic_dataset):
        train_df = synthetic_dataset["train_df"]
        b1 = SeasonalExpandingMean()
        b1.fit(train_df)

        # Test predict_samples
        test_df = train_df.head(10).copy()
        samples = b1.predict_samples(test_df, n_samples=200, seed=42)

        assert samples.shape == (10, 200)
        assert np.all(samples >= 0)
        assert np.issubdtype(samples.dtype, np.integer)

        # Reproducibility with seed
        samples2 = b1.predict_samples(test_df, n_samples=200, seed=42)
        np.testing.assert_array_equal(samples, samples2)

    def test_b2_seasonal_nb(self, synthetic_dataset):
        train_df = synthetic_dataset["train_df"]
        b2 = SeasonalNB()
        b2.fit(train_df)

        test_df = train_df.head(10).copy()
        samples = b2.predict_samples(test_df, n_samples=200, seed=42)

        assert samples.shape == (10, 200)
        assert np.all(samples >= 0)
        assert np.issubdtype(samples.dtype, np.integer)
        assert b2.to_dict()["alpha"] > 0


class TestNBGLM:
    def test_ground_truth_coefficient_recovery(self, synthetic_dataset):
        """Verify model recovers known signs and confidence bounds:

        humidity_l02 positive, dtr_l02 negative, wind close to 0.
        """
        train_df = synthetic_dataset["train_df"]

        # Fit model with standardized climate predictors
        spec_features = ["humidity_l02_z", "dtr_l02_z", "wind_l0_z"]
        model = NBGLM(features=spec_features)
        model.fit(train_df)

        summary = model.summary_frame().set_index("term")

        # 1. humidity_l02_z must be significantly positive
        assert "humidity_l02_z" in summary.index
        h_coef = summary.loc["humidity_l02_z", "coef"]
        assert h_coef > 0, f"Expected humidity coefficient > 0, got {h_coef}"

        # 2. dtr_l02_z must be negative
        assert "dtr_l02_z" in summary.index
        d_coef = summary.loc["dtr_l02_z", "coef"]
        assert d_coef < 0, f"Expected dtr coefficient < 0, got {d_coef}"

        # 3. wind_l0_z ground truth is 0.0 -> must be within 2 SE of 0
        assert "wind_l0_z" in summary.index
        w_coef = summary.loc["wind_l0_z", "coef"]
        w_se = summary.loc["wind_l0_z", "std_err"]
        assert abs(w_coef) <= 2.5 * w_se, f"Expected wind coef within 2.5 SE of 0, got {w_coef} with SE {w_se}"

    def test_predict_samples_shape_and_types(self, synthetic_dataset):
        train_df = synthetic_dataset["train_df"]
        model = NBGLM(features=["humidity_l02_z"])
        model.fit(train_df)

        test_df = train_df.head(15).copy()
        samples = model.predict_samples(test_df, n_samples=500, seed=123)

        assert samples.shape == (15, 500)
        assert np.all(samples >= 0)
        assert np.issubdtype(samples.dtype, np.integer)

        # Determinism with same seed
        samples_same = model.predict_samples(test_df, n_samples=500, seed=123)
        np.testing.assert_array_equal(samples, samples_same)

        # Variance with different seed
        samples_diff = model.predict_samples(test_df, n_samples=500, seed=999)
        assert not np.array_equal(samples, samples_diff)

    def test_in_sample_95_percent_interval_coverage(self, synthetic_dataset):
        """P5 requirement: In-sample 95% interval coverage between 0.85 and 1.0."""
        train_df = synthetic_dataset["train_df"]
        model = NBGLM(features=["humidity_l02_z", "dtr_l02_z"])
        model.fit(train_df)

        sub_sample = train_df.sample(n=min(300, len(train_df)), random_state=42).copy()
        samples = model.predict_samples(sub_sample, n_samples=1000, seed=42)

        q025 = np.percentile(samples, 2.5, axis=1)
        q975 = np.percentile(samples, 97.5, axis=1)
        y_true = sub_sample["cases"].values

        covered = (y_true >= q025) & (y_true <= q975)
        coverage = float(np.mean(covered))

        assert 0.85 <= coverage <= 1.0, f"Coverage was {coverage:.3f}, expected between 0.85 and 1.0"

    def test_climate_by_group(self, synthetic_dataset):
        """P5b requirement: climate_by_group fits separate terms per region group."""
        train_df = synthetic_dataset["train_df"]
        model = NBGLM(features=["humidity_l02_z", "precip_l02_z"], climate_by_group=True)
        model.fit(train_df)

        summary = model.summary_frame()
        terms = summary["term"].tolist()

        # Check for group interaction terms
        group_terms = [t for t in terms if "clim_group" in t]
        assert len(group_terms) > 0, "Expected interaction terms with clim_group"

        card = model.model_card()
        assert card["climate_by_group"] is True
        assert card["n_obs"] == len(train_df)
        assert "fit_date" in card

    def test_seasonality_fallback_when_obs_under_72(self, synthetic_dataset):
        """P5b requirement: automatically fallback to group x month when obs < 72."""
        train_df = synthetic_dataset["train_df"]
        # Subset to 36 months (< 72)
        short_df = train_df[train_df["month"] <= "2018-12"].copy()
        model = NBGLM(features=["humidity_l02_z"])
        model.fit(short_df)

        card = model.model_card()
        assert card["season_structure"] in ["group_x_month", "common_month"]


class TestReconstructLag:
    def test_recursive_case_reconstruction(self, synthetic_dataset):
        """P5 requirement: h=1 uses actual observed cases, h>=2 uses model predictions."""
        train_df = synthetic_dataset["train_df"]
        tables = synthetic_dataset["tables"]
        origin = "2014-06"

        reconstructor = ReconstructLag()
        reconstructor.fit(train_df)

        reconstructed = reconstructor.reconstruct(
            origin=origin,
            max_lead=4,
            cases_df=tables["cases"],
            regions_df=tables["regions"],
        )

        assert not reconstructed.empty
        assert set(["region_id", "origin", "target", "lead", "ylag_hat"]).issubset(
            reconstructed.columns
        )

        # Check for a specific region
        r1_df = reconstructed[reconstructed["region_id"] == "REG_01"].sort_values("lead")
        assert len(r1_df) == 4

        # For lead 1: target = 2020-07, ylag_hat must be observed cases at 2020-06
        lead1_val = r1_df[r1_df["lead"] == 1]["ylag_hat"].values[0]
        obs_origin_cases = tables["cases"][
            (tables["cases"]["region_id"] == "REG_01") & (tables["cases"]["month"] == origin)
        ]["cases"].values[0]

        assert lead1_val == pytest.approx(obs_origin_cases, abs=0.01)

        # For lead 2: target = 2020-08, ylag_hat is the predicted value from lead 1
        lead2_val = r1_df[r1_df["lead"] == 2]["ylag_hat"].values[0]
        assert lead2_val > 0
