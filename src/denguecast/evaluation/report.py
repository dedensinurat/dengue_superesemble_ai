"""Evaluation report generation and province eligibility gate (MODEL_SPEC C.11 & C.13).

Comprehensive evaluation across walk-forward splits:
- Probabilistic scores (CRPS, CRPSS vs B1/B2)
- Point metrics (MAE, RMSE, Bias)
- Interval metrics (95% empirical coverage, interval width)
- P7b Indonesian eligibility gate: median CRPSS > 0 and CI_low > 0 per province
- P7b Sensitivity analysis: with and without 2020-2021 COVID-19 period
"""

from __future__ import annotations

import logging
import os
import uuid
from typing import Any

import numpy as np
import pandas as pd

from denguecast.evaluation.scores import (
    coverage_95,
    crps_samples,
    crpss,
    interval_width_95,
)
from denguecast.evaluation.splits import ExpandingWindowSplitter
from denguecast.features.build import build_training_frame
from denguecast.models.base import ForecastModel, ModelSpec
from denguecast.models.baseline import SeasonalExpandingMean, SeasonalNB
from denguecast.models.registry import build_model

logger = logging.getLogger(__name__)


def compute_province_eligibility(
    eval_df: pd.DataFrame,
    n_bootstraps: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compute P7b Province Eligibility Gate.

    Rule:
    A province is 'eligible' (layak) if median CRPSS on leads 1..3 against B1 > 0
    AND the lower bound of the 95% bootstrap confidence interval > 0.
    Otherwise, the province is flagged as 'kurang andal' (reliable = False).

    Args:
        eval_df: DataFrame with columns [region_id, lead, crpss].
        n_bootstraps: Number of bootstrap resamples.
        seed: Random seed.

    Returns:
        DataFrame [region_id, crpss_median, ci_low, ci_high, eligible, status].
    """
    rng = np.random.default_rng(seed)

    # Filter for leads 1 to 3
    sub = eval_df[eval_df["lead"].isin([1, 2, 3])].copy()
    if sub.empty:
        sub = eval_df.copy()

    records = []
    for rid, grp in sub.groupby("region_id"):
        vals = grp["crpss"].dropna().values
        if len(vals) == 0:
            records.append(
                {
                    "region_id": str(rid),
                    "crpss_median": 0.0,
                    "ci_low": -1.0,
                    "ci_high": 0.0,
                    "eligible": False,
                    "status": "kurang andal",
                }
            )
            continue

        median_val = float(np.median(vals))
        boot_medians = []
        for _ in range(n_bootstraps):
            sample_vals = rng.choice(vals, size=len(vals), replace=True)
            boot_medians.append(float(np.median(sample_vals)))

        ci_low = float(np.percentile(boot_medians, 2.5)) if boot_medians else median_val
        ci_high = float(np.percentile(boot_medians, 97.5)) if boot_medians else median_val

        eligible = (median_val > 0.0) and (ci_low > 0.0)
        status = "layak" if eligible else "kurang andal"

        records.append(
            {
                "region_id": str(rid),
                "crpss_median": round(median_val, 4),
                "ci_low": round(ci_low, 4),
                "ci_high": round(ci_high, 4),
                "eligible": bool(eligible),
                "status": status,
            }
        )

    return pd.DataFrame(records).sort_values("crpss_median", ascending=False).reset_index(drop=True)


def run_full_evaluation(
    data: dict[str, pd.DataFrame] | pd.DataFrame,
    models: list[ForecastModel] | None = None,
    splitter: Any = None,
    config: Any = None,
    exclude_periods: list[str] | None = None,
    n_samples: int = 200,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full expanding-window evaluation and compute eligibility.

    Args:
        data: Dict of data tables or pre-built training frame.
        models: Optional list of fitted models or specs to evaluate.
        splitter: ExpandingWindowSplitter instance.
        config: DengueCastConfig instance.
        exclude_periods: Optional list of years/months to exclude (e.g. ['2020', '2021']).
        n_samples: Number of forecast samples per evaluation point.

    Returns:
        Tuple of (eval_scores_df, province_eligibility_df).
    """
    if isinstance(data, pd.DataFrame):
        train_df = data.copy()
    elif isinstance(data, dict):
        if "train_df" in data:
            train_df = data["train_df"].copy()
        else:
            train_df, _, _ = build_training_frame(
                cases=data["cases"],
                pop=data["population"],
                climate_obs=data["climate_obs"],
                enso=data["enso"],
                regions=data.get("regions"),
                config=config,
            )
    else:
        raise TypeError(f"Unsupported data format: {type(data)}")

    # Apply exclude_periods if requested
    if exclude_periods:
        for p in exclude_periods:
            train_df = train_df[~train_df["month"].str.startswith(p)].copy()

    all_months = sorted(train_df["month"].unique().tolist())

    if splitter is None:
        init_m = min(60, max(24, len(all_months) - 12))
        step_m = max(3, len(all_months) // 8)
        splitter = ExpandingWindowSplitter(initial_train_months=init_m, step=step_m, max_lead=6)

    splits = splitter.split(all_months)
    if not splits:
        logger.warning("No splits formed for full evaluation.")
        return pd.DataFrame(), pd.DataFrame()

    run_id = f"eval_{uuid.uuid4().hex[:8]}"

    # Setup models: Baselines B1 & B2 + Default NBGLM if none supplied
    eval_models: list[ForecastModel] = []
    if models is not None and len(models) > 0:
        eval_models = models
    else:
        b1 = SeasonalExpandingMean()
        b2 = SeasonalNB()
        # Find active standardized features
        feature_cols = [c for c in ["humidity_l02_z", "dtr_l02_z", "wind_l0_z"] if c in train_df.columns]
        if not feature_cols:
            feature_cols = [c for c in ["humidity_l02", "dtr_l02"] if c in train_df.columns]
        nb = build_model(ModelSpec(model_id="nbglm_eval", features=feature_cols))
        eval_models = [b1, b2, nb]

    records = []
    crpss_tracker = []

    for sp in splits:
        train_split = train_df[train_df["month"].isin(sp.train_months)].copy()
        if train_split.empty:
            continue

        # Fit all models on training window
        for m in eval_models:
            try:
                m.fit(train_split)
            except Exception as e:
                logger.debug(f"Model fit failed for {m.spec.model_id}: {e}")

        for target_month, lead in sp.test_targets:
            test_sub = train_df[train_df["month"] == target_month].copy()
            if test_sub.empty:
                continue

            y_obs = test_sub["cases"].values
            regions = test_sub["region_id"].values

            # Generate samples for all models
            model_preds = {}
            for m in eval_models:
                try:
                    s = m.predict_samples(test_sub, n_samples=n_samples, seed=42 + lead)
                    model_preds[m.spec.model_id] = s
                except Exception:
                    pass

            b1_samples = model_preds.get("B1_seasonal_mean", model_preds.get(eval_models[0].spec.model_id))

            for mid, s_mat in model_preds.items():
                crps_arr = crps_samples(s_mat, y_obs)
                b1_crps_arr = crps_samples(b1_samples, y_obs) if b1_samples is not None else crps_arr

                for idx, rid in enumerate(regions):
                    c_val = float(crps_arr[idx]) if isinstance(crps_arr, np.ndarray) else float(crps_arr)
                    b_val = float(b1_crps_arr[idx]) if isinstance(b1_crps_arr, np.ndarray) else float(b1_crps_arr)
                    c_skill = crpss(c_val, b_val)
                    m_val = float(np.mean(s_mat[idx]))
                    y_val = float(y_obs[idx])

                    records.append(
                        {
                            "run_id": run_id,
                            "model_id": mid,
                            "region_id": str(rid),
                            "origin": sp.origin,
                            "target_month": target_month,
                            "lead": lead,
                            "crps": c_val,
                            "crpss": c_skill,
                            "mae": abs(m_val - y_val),
                            "rmse": (m_val - y_val) ** 2,
                            "bias": m_val - y_val,
                            "coverage_95": float(coverage_95(s_mat[idx], y_val)),
                            "interval_width_95": float(interval_width_95(s_mat[idx])),
                        }
                    )
                    if mid not in ["B1_seasonal_mean", "B2_seasonal_nb"]:
                        crpss_tracker.append(
                            {"region_id": str(rid), "lead": lead, "crpss": c_skill}
                        )

    scores_df = pd.DataFrame(records)
    crpss_df = pd.DataFrame(crpss_tracker)
    eligibility_df = compute_province_eligibility(crpss_df) if not crpss_df.empty else pd.DataFrame()

    return scores_df, eligibility_df


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Format DataFrame as Markdown table without external tabulate dependency."""
    if df.empty:
        return ""
    headers = list(df.columns)
    header_line = "| " + " | ".join(str(h) for h in headers) + " |"
    sep_line = "| " + " | ".join("---" for _ in headers) + " |"
    data_lines = []
    for _, row in df.iterrows():
        row_str = []
        for h in headers:
            val = row[h]
            if isinstance(val, float | np.floating):
                row_str.append(f"{val:.3f}")
            else:
                row_str.append(str(val))
        data_lines.append("| " + " | ".join(row_str) + " |")
    return "\n".join([header_line, sep_line] + data_lines)


def generate_report(
    scores_df: pd.DataFrame,
    eligibility_df: pd.DataFrame | None = None,
    output_path: str = "reports/evaluation_report.md",
    sensitivity_diff: dict[str, Any] | None = None,
) -> str:
    """Generate Markdown evaluation report with Indonesia-specific sections."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    lines = []
    lines.append("# DengueCast — Laporan Evaluasi Model Peramalan DBD Probabilistik")
    lines.append("")
    lines.append(
        "> **Pemberitahuan Penting**: DengueCast adalah prototipe riset untuk peramalan DBD probabilistik. "
        "Bukan alat keputusan klinis atau kebijakan. Data kasus berupa DBD berdiagnosis klinis."
    )
    lines.append("")

    if scores_df.empty:
        lines.append("*Tidak ada skor evaluasi yang tersedia.*")
        report_text = "\n".join(lines)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        return output_path

    # 1. Ringkasan Kinerja Keseluruhan
    lines.append("## 1. Ringkasan Kinerja Keseluruhan per Model")
    lines.append("")
    summary = (
        scores_df.groupby("model_id")
        .agg(
            CRPS=("crps", "mean"),
            CRPSS_vs_B1=("crpss", "median"),
            MAE=("mae", "mean"),
            RMSE=("rmse", lambda x: float(np.sqrt(np.mean(x)))),
            Cakupan_95=("coverage_95", "mean"),
            Lebar_Interval=("interval_width_95", "mean"),
        )
        .round(3)
        .reset_index()
    )
    lines.append(_df_to_markdown(summary))
    lines.append("")

    # 2. Kinerja per Lead Horizon (Lead 1 sampai 6)
    lines.append("## 2. Kinerja per Lead Horizon (1 - 6 Bulan)")
    lines.append("")
    lead_summary = (
        scores_df.groupby(["model_id", "lead"])
        .agg(
            CRPS=("crps", "mean"),
            CRPSS_Median=("crpss", "median"),
            Cakupan_95=("coverage_95", "mean"),
        )
        .round(3)
        .reset_index()
    )
    lines.append(_df_to_markdown(lead_summary))
    lines.append("")

    # 3. Gerbang Kelayakan Provinsi (P7b)
    lines.append("## 3. Gerbang Kelayakan Provinsi (P7b)")
    lines.append("")
    lines.append(
        "Provinsi dinyatakan **layak** jika median CRPSS pada lead 1-3 > 0 dan batas bawah 95% interval bootstrap > 0. "
        "Provinsi di luar kriteria ini ditandai sebagai **kurang andal** (`reliable = false`)."
    )
    lines.append("")

    if eligibility_df is not None and not eligibility_df.empty:
        n_eligible = int(eligibility_df["eligible"].sum())
        total_prov = len(eligibility_df)
        lines.append(
            f"**Ringkasan Status**: {n_eligible} dari {total_prov} provinsi dinyatakan **layak**."
        )
        lines.append("")
        lines.append(_df_to_markdown(eligibility_df))
    else:
        lines.append("*Tabel kelayakan belum dihitung.*")
    lines.append("")

    # 4. Analisis Sensitivitas (P7b)
    lines.append("## 4. Analisis Sensitivitas: Pengaruh Pandemi COVID-19 (2020-2021)")
    lines.append("")
    if sensitivity_diff:
        lines.append("| Kondisi | Mean CRPS | Median CRPSS | Cakupan 95% |")
        lines.append("|---|---|---|---|")
        for k, v in sensitivity_diff.items():
            lines.append(f"| {k} | {v.get('crps', 0.0):.3f} | {v.get('crpss', 0.0):.3f} | {v.get('coverage', 0.0):.3f} |")
    else:
        lines.append("Analisis sensitivitas dilakukan dengan membandingkan evaluasi lengkap dengan evaluasi yang mengecualikan periode anomali 2020-2021.")
    lines.append("")

    report_text = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Generated evaluation report at {output_path}")
    return output_path
