"""Unit tests for evaluation metrics, splits, classification, reliability, and reports (P7 & P7b)."""

import os

import numpy as np
import pandas as pd
import pytest
import scipy.stats as stats

from denguecast.data.synthetic import generate_synthetic_data
from denguecast.evaluation.classification import (
    brier_score,
    compute_auc_with_ci,
    contingency_table,
)
from denguecast.evaluation.reliability import reliability_table
from denguecast.evaluation.report import (
    compute_province_eligibility,
    generate_report,
    run_full_evaluation,
)
from denguecast.evaluation.scores import (
    coverage_95,
    crps_samples,
    crpss,
    interval_width_95,
    mae,
    rmse,
)
from denguecast.evaluation.splits import ExpandingWindowSplitter
from denguecast.features.build import build_training_frame


class TestExpandingWindowSplitter:
    def test_splitter_never_leaks_future(self):
        """Training months must strictly be <= origin."""
        months = [f"2010-{m:02d}" for m in range(1, 13)] + [f"2011-{m:02d}" for m in range(1, 13)]
        splitter = ExpandingWindowSplitter(initial_train_months=12, step=3, max_lead=4)
        splits = splitter.split(months)

        assert len(splits) > 0
        for sp in splits:
            for tm in sp.train_months:
                assert tm <= sp.origin, f"Leakage: train month {tm} > origin {sp.origin}"
            for tgt, _lead in sp.test_targets:
                assert tgt > sp.origin, f"Test target {tgt} must be > origin {sp.origin}"


class TestScoresAndAnalyticalCRPS:
    def test_crps_against_analytical_normal(self):
        """Sample CRPS from normal distribution matches closed-form analytical CRPS."""
        mu, sigma, y = 10.0, 3.0, 12.0

        # Closed-form analytical CRPS for Gaussian (Gneiting et al.)
        z = (y - mu) / sigma
        exact_crps = float(
            sigma
            * (
                z * (2 * stats.norm.cdf(z) - 1)
                + 2 * stats.norm.pdf(z)
                - 1.0 / np.sqrt(np.pi)
            )
        )

        # Draw large sample
        rng = np.random.default_rng(42)
        samples = rng.normal(mu, sigma, size=100_000)
        est_crps = float(crps_samples(samples, y))

        assert est_crps == pytest.approx(exact_crps, abs=0.03)

    def test_crps_vectorized_2d(self):
        rng = np.random.default_rng(42)
        samples = rng.poisson(20, size=(10, 500))
        y = np.array([18, 22, 19, 25, 20, 21, 15, 24, 20, 19])

        vec_crps = crps_samples(samples, y)
        assert len(vec_crps) == 10

        # Row by row match
        for i in range(10):
            single_crps = crps_samples(samples[i], y[i])
            assert vec_crps[i] == pytest.approx(single_crps, abs=1e-6)

    def test_crpss_values(self):
        assert crpss(0.0, 10.0) == 1.0
        assert crpss(10.0, 10.0) == 0.0
        assert crpss(12.0, 10.0) == pytest.approx(-0.2)

    def test_point_and_interval_metrics(self):
        y_true = np.array([10, 20, 30])
        y_pred = np.array([12, 18, 30])

        assert mae(y_true, y_pred) == pytest.approx(4.0 / 3.0)
        assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(8.0 / 3.0))

        samples = np.array([[8, 10, 12], [18, 20, 22], [28, 30, 32]])
        assert coverage_95(samples, y_true) == 1.0
        assert interval_width_95(samples) > 0


class TestClassificationAndReliability:
    def test_contingency_table_metrics(self):
        p_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
        y_true = np.array([1, 1, 0, 1, 0, 0])

        table = contingency_table(p_prob, y_true, cutoff=0.5)
        # Hits: p>=0.5 & y==1 -> (0.9, 1), (0.8, 1) -> 2
        assert table["hits"] == 2
        # False alarms: p>=0.5 & y==0 -> (0.7, 0) -> 1
        assert table["false_alarms"] == 1
        # Misses: p<0.5 & y==1 -> (0.3, 1) -> 1
        assert table["misses"] == 1
        # Correct rejections: p<0.5 & y==0 -> (0.2, 0), (0.1, 0) -> 2
        assert table["correct_rejections"] == 2

        assert table["sensitivity"] == pytest.approx(2.0 / 3.0, abs=1e-3)
        assert table["ppv"] == pytest.approx(2.0 / 3.0, abs=1e-3)
        assert table["accuracy"] == pytest.approx(4.0 / 6.0, abs=1e-3)
        assert brier_score(p_prob, y_true) > 0

    def test_auc_with_confidence_interval(self):
        p_prob = np.array([0.9, 0.8, 0.7, 0.6, 0.4, 0.3, 0.2, 0.1])
        y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])

        auc_info = compute_auc_with_ci(p_prob, y_true, n_bootstraps=50)
        assert auc_info["auc"] == 1.0
        assert auc_info["ci_lower"] <= 1.0

    def test_reliability_table_bins(self):
        p_prob = np.linspace(0.05, 0.95, 20)
        y_true = (p_prob > 0.5).astype(int)

        rel = reliability_table(p_prob, y_true, n_bins=10)
        assert len(rel) == 10
        assert set(["bin_lower", "bin_upper", "mean_forecast_prob", "observed_freq", "count"]).issubset(
            rel.columns
        )
        assert rel["count"].sum() == 20


class TestProvinceEligibilityAndReport:
    def test_province_eligibility_criteria(self):
        """Province is eligible iff median CRPSS > 0 and lower CI > 0."""
        eval_data = pd.DataFrame(
            [
                {"region_id": "P_GOOD", "lead": 1, "crpss": 0.25},
                {"region_id": "P_GOOD", "lead": 2, "crpss": 0.30},
                {"region_id": "P_GOOD", "lead": 3, "crpss": 0.28},
                {"region_id": "P_BAD", "lead": 1, "crpss": -0.15},
                {"region_id": "P_BAD", "lead": 2, "crpss": -0.05},
                {"region_id": "P_BAD", "lead": 3, "crpss": 0.02},
            ]
        )

        elig = compute_province_eligibility(eval_data, n_bootstraps=100)
        assert len(elig) == 2

        good_row = elig[elig["region_id"] == "P_GOOD"].iloc[0]
        assert good_row["eligible"] is True or good_row["eligible"] == 1
        assert good_row["status"] == "layak"

        bad_row = elig[elig["region_id"] == "P_BAD"].iloc[0]
        assert bad_row["eligible"] is False or bad_row["eligible"] == 0
        assert bad_row["status"] == "kurang andal"

    def test_full_evaluation_and_report_generation(self, tmp_path):
        """End-to-end test on small synthetic dataset."""
        tables = generate_synthetic_data(n_regions=4, n_years=4, seed=42)
        train_df, _, _ = build_training_frame(
            cases=tables["cases"],
            pop=tables["population"],
            climate_obs=tables["climate_obs"],
            enso=tables["enso"],
            regions=tables["regions"],
            origin="2013-12",
        )

        splitter = ExpandingWindowSplitter(initial_train_months=36, step=6, max_lead=2)
        scores_df, elig_df = run_full_evaluation(train_df, splitter=splitter, n_samples=50)

        assert not scores_df.empty
        assert "crps" in scores_df.columns
        assert not elig_df.empty

        out_file = str(tmp_path / "eval_report.md")
        report_path = generate_report(scores_df, elig_df, output_path=out_file)

        assert os.path.exists(report_path)
        with open(report_path, encoding="utf-8") as f:
            content = f.read()

        assert "Laporan Evaluasi" in content
        assert "Gerbang Kelayakan" in content
        assert "kurang andal" in content or "layak" in content
