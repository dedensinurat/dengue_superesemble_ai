"""Unit tests for candidates, selection, and superensemble stacking (P6)."""

import numpy as np
import pandas as pd
import pytest

from denguecast.data.synthetic import generate_synthetic_data
from denguecast.ensemble.candidates import generate_candidates
from denguecast.ensemble.selection import score_candidates, select_top_k
from denguecast.ensemble.stacking import Superensemble, compute_weights
from denguecast.evaluation.scores import crps_samples
from denguecast.features.build import build_training_frame
from denguecast.models.base import ModelSpec
from denguecast.models.baseline import SeasonalExpandingMean, SeasonalNB
from denguecast.models.nb_glm import NBGLM


@pytest.fixture(scope="module")
def small_train_data():
    """Deterministic small synthetic dataset for fast ensemble testing."""
    tables = generate_synthetic_data(n_regions=8, n_years=5, seed=42)
    train_df, scaler, features = build_training_frame(
        cases=tables["cases"],
        pop=tables["population"],
        climate_obs=tables["climate_obs"],
        enso=tables["enso"],
        regions=tables["regions"],
        origin="2014-12",
    )
    return {
        "tables": tables,
        "train_df": train_df,
        "features": features,
    }


class TestCandidateGeneration:
    def test_candidates_deterministic_and_unique(self):
        """Candidates must be deterministically ordered and unique."""
        c1 = generate_candidates(available_features=["humidity_l02_z", "dtr_l02_z", "wind_l0_z"])
        c2 = generate_candidates(available_features=["humidity_l02_z", "dtr_l02_z", "wind_l0_z"])

        assert len(c1) > 0
        assert [s.model_id for s in c1] == [s.model_id for s in c2]

        mids = [s.model_id for s in c1]
        assert len(mids) == len(set(mids)), "Candidate model_ids must be unique"

    def test_candidate_mutual_exclusion(self):
        """tmin and tmax must never be selected together with tmean or dtr."""
        candidates = generate_candidates(
            available_features=[
                "tmean_l02_z",
                "tmin_l02_z",
                "tmax_l02_z",
                "dtr_l02_z",
                "humidity_l02_z",
            ]
        )
        for c in candidates:
            feats = c.features
            has_tdual = "tmin_l02_z" in feats or "tmax_l02_z" in feats
            has_tm_or_dtr = "tmean_l02_z" in feats or "dtr_l02_z" in feats
            assert not (has_tdual and has_tm_or_dtr), f"Violation in spec {c.model_id}: {feats}"


class TestStackingAndWeights:
    def test_weights_sum_to_one_and_monotone(self):
        """Weights must sum to 1.0 and decrease as CRPS increases."""
        crps_dict = {
            "model_best": 1.5,
            "model_mid": 2.5,
            "model_worst": 5.0,
        }
        weights = compute_weights(crps_dict)

        assert np.isclose(weights.sum(), 1.0)
        assert weights["model_best"] > weights["model_mid"] > weights["model_worst"]

    def test_superensemble_exact_sample_count(self, small_train_data):
        """Linear pool must produce exactly n_samples draws with correct shape."""
        train_df = small_train_data["train_df"]

        m1 = SeasonalExpandingMean().fit(train_df)
        m2 = SeasonalNB().fit(train_df)
        m3 = NBGLM(features=["humidity_l02_z"]).fit(train_df)

        weights = pd.Series([0.5, 0.3, 0.2], index=[m1.spec.model_id, m2.spec.model_id, m3.spec.model_id])
        ensemble = Superensemble([m1, m2, m3], weights)

        test_df = train_df.head(10).copy()
        n_samples = 1000
        samples = ensemble.predict_samples(test_df, n_samples=n_samples, seed=42)

        assert samples.shape == (10, n_samples)
        assert np.all(samples >= 0)
        assert np.issubdtype(samples.dtype, np.integer)

    def test_superensemble_crps_improvement(self, small_train_data):
        """CRPS of ensemble should be <= mean CRPS of constituent members on test set."""
        train_df = small_train_data["train_df"]

        split_month = "2014-06"
        train_sub = train_df[train_df["month"] <= split_month].copy()
        test_sub = train_df[train_df["month"] > split_month].copy()

        m1 = SeasonalExpandingMean().fit(train_sub)
        m2 = SeasonalNB().fit(train_sub)
        m3 = NBGLM(features=["humidity_l02_z", "dtr_l02_z"]).fit(train_sub)

        y_test = test_sub["cases"].values

        s1 = m1.predict_samples(test_sub, n_samples=300, seed=1)
        s2 = m2.predict_samples(test_sub, n_samples=300, seed=2)
        s3 = m3.predict_samples(test_sub, n_samples=300, seed=3)

        crps1 = float(np.mean(crps_samples(s1, y_test)))
        crps2 = float(np.mean(crps_samples(s2, y_test)))
        crps3 = float(np.mean(crps_samples(s3, y_test)))

        weights = compute_weights({
            m1.spec.model_id: crps1,
            m2.spec.model_id: crps2,
            m3.spec.model_id: crps3,
        })

        ensemble = Superensemble([m1, m2, m3], weights)
        s_ens = ensemble.predict_samples(test_sub, n_samples=300, seed=42)
        crps_ens = float(np.mean(crps_samples(s_ens, y_test)))

        avg_member_crps = (crps1 + crps2 + crps3) / 3.0
        assert crps_ens <= avg_member_crps + 0.05, f"Ensemble ({crps_ens}) should not exceed mean member CRPS ({avg_member_crps})"


class TestCandidateScoringAndSelection:
    def test_score_and_select_top_k(self, small_train_data):
        train_df = small_train_data["train_df"]

        candidates = [
            ModelSpec("nbglm_h", features=["humidity_l02_z"]),
            ModelSpec("nbglm_d", features=["dtr_l02_z"]),
            ModelSpec("nbglm_hd", features=["humidity_l02_z", "dtr_l02_z"]),
        ]

        scores = score_candidates(candidates, train_df)
        assert not scores.empty
        assert "model_id" in scores.columns
        assert "crps_mean" in scores.columns

        selected = select_top_k(scores, k=2, available_specs=candidates)
        mids = [s.model_id for s in selected]

        # Must always include baselines
        assert "B1_seasonal_mean" in mids
        assert "B2_seasonal_nb" in mids
        # Plus 2 candidates -> total 4
        assert len(selected) == 4
