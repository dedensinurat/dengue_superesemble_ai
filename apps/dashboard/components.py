"""Dashboard UI components and Plotly charts for DengueCast (P10)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Color-blind safe palette
COLOR_MAP = {
    "Rendah": "#2b83ba",    # Blue
    "Waspada": "#abdda4",   # Soft Green-Teal
    "Siaga": "#fdae61",     # Warm Amber-Orange
    "Awas": "#d7191c",      # Vivid Red
}

RISK_ORDER = ["Rendah", "Waspada", "Siaga", "Awas"]


def render_synthetic_banner() -> None:
    """Render prominent synthetic data warning banner."""
    st.error(
        "⚠️ **PERHATIAN: DATA SINTETIS (RESEARCH PROTOTYPE)**\n\n"
        "Seluruh data epidemiologi, iklim, dan prakiraan yang ditampilkan pada dashboard ini "
        "dihasilkan secara sintetis untuk keperluan riset dan demonstrasi metode ensemble D-MOSS. "
        "**Bukan data observasi riil BMKG/Kemenkes** dan **BUKAN ALAT KEPUTUSAN KLINIS ATAU KEBIJAKAN**.",
        icon="⚠️",
    )


def render_reliability_banner(reliable: bool, region_name: str = "") -> None:
    """Render province eligibility and reliability alert."""
    if not reliable:
        st.warning(
            f"⚠️ **STATUS KEANDALAN: KURANG ANDAL** ({region_name or 'Wilayah ini'})\n\n"
            "Berdasarkan analisis evaluasi riwayat (expanding-window cross-validation), "
            "provinsi ini tidak memenuhi gerbang kelayakan (median CRPSS $\\le 0$ atau batas "
            "bawah CI $\\le 0$). Prakiraan iklim belum terbukti mengungguli baseline musiman historis.",
            icon="ℹ️",
        )
    else:
        st.success(
            f"✅ **STATUS KEANDALAN: LAYAK** ({region_name or 'Wilayah ini'})\n\n"
            "Provinsi ini memenuhi gerbang kelayakan evaluasi (median CRPSS > 0 dan interval "
            "bawah bootstrap > 0 mengungguli baseline musiman).",
            icon="✅",
        )


def render_disclaimer() -> None:
    """Render clinical and legal disclaimer."""
    st.caption(
        "**Pemberitahuan Resmi**: DengueCast adalah prototipe riset sistem peramalan demam berdarah dengue (DBD) "
        "probabilistik terinspirasi D-MOSS (Colón-González et al. 2021; Campbell et al. 2026). "
        "Kasus mengacu pada DBD berdiagnosis klinis. Sistem ini tidak memberikan diagnosis medis individu "
        "atau arahan intervensi operasional langsung tanpa verifikasi otoritas kesehatan setempat."
    )


def create_risk_map(
    forecasts_df: pd.DataFrame,
    regions_df: pd.DataFrame,
    lead: int = 1,
    threshold: str = "p_exceed_m1sd",
) -> go.Figure:
    """Render interactive risk map for Indonesia administrative regions."""
    sub_fc = forecasts_df[forecasts_df["lead"] == lead].copy()

    if not regions_df.empty and "region_id" in regions_df.columns:
        merged = pd.merge(sub_fc, regions_df, on="region_id", how="left")
    else:
        merged = sub_fc.copy()
        if "lat" not in merged:
            merged["lat"] = -2.5
        if "lon" not in merged:
            merged["lon"] = 118.0

    # Ensure coordinates
    merged["lat"] = merged["lat"].fillna(-2.5)
    merged["lon"] = merged["lon"].fillna(118.0)
    merged["name"] = merged["name"].fillna(merged["region_id"])

    threshold_label = {
        "p_exceed_m1sd": "P(Kasus > Rata-rata + 1 SD)",
        "p_exceed_m2sd": "P(Kasus > Rata-rata + 2 SD)",
        "p_exceed_p75": "P(Kasus > Persentil 75)",
        "p_exceed_p95": "P(Kasus > Persentil 95)",
    }.get(threshold, threshold)

    fig = px.scatter_geo(
        merged,
        lat="lat",
        lon="lon",
        color="risk_level",
        color_discrete_map=COLOR_MAP,
        category_orders={"risk_level": RISK_ORDER},
        size=np.maximum(10, np.sqrt(merged["mean"].fillna(10))),
        hover_name="name",
        hover_data={
            "region_id": True,
            "risk_level": True,
            threshold: ":.2f",
            "q50": ":.1f",
            "reliable": True,
            "lat": False,
            "lon": False,
        },
        labels={
            "risk_level": "Tingkat Risiko",
            threshold: threshold_label,
            "q50": "Median Kasus",
            "reliable": "Status Andal",
        },
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
        showcountries=True,
        countrycolor="#cccccc",
        showcoastlines=True,
        coastlinecolor="#bbbbbb",
        showland=True,
        landcolor="#f5f5f5",
        showocean=True,
        oceancolor="#eaf2f8",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="",
        ),
        height=480,
    )
    return fig


REAL_COLOR_MAP = {
    "Awas": "#d7191c",      # Vivid Red
    "Siaga": "#fdae61",     # Warm Amber-Orange
    "Waspada": "#abdda4",   # Soft Green-Teal
    "Rendah": "#2b83ba",    # Slate Blue
}

REAL_CATEGORY_ORDER = [
    "Awas",
    "Siaga",
    "Waspada",
    "Rendah",
]


def create_real_dengue_map(
    summary_df: pd.DataFrame,
    metric_choice: str = "total_cases",
    geojson_path: str = "data/ref/indonesia_provinces.geojson",
) -> go.Figure:
    """Render real Indonesia dengue distribution map using official boundaries."""
    geo_file = Path(geojson_path)
    cat_col = "category_total"
    val_col = "total_cases"
    val_title = "Total Kasus Kumulatif (2000-2024)"

    if metric_choice == "cases_2024":
        cat_col = "category_2024"
        val_col = "cases_2024"
        val_title = "Kasus Tahun 2024 (Outbreak Terkini)"
    elif metric_choice == "incidence_rate":
        cat_col = "category_ir"
        val_col = "incidence_rate"
        val_title = "Angka Insidens (IR per 100k Penduduk)"

    if geo_file.exists():
        with open(geo_file, encoding="utf-8") as f:
            geojson_data = json.load(f)

        fig = px.choropleth_map(
            summary_df,
            geojson=geojson_data,
            locations="region_id",
            featureidkey="id",
            color=cat_col,
            color_discrete_map=REAL_COLOR_MAP,
            category_orders={cat_col: REAL_CATEGORY_ORDER},
            hover_name="name",
            hover_data={
                "region_id": True,
                val_col: True,
                "total_cases": ":,",
                "cases_2024": ":,",
                "incidence_rate": ":.1f",
                "national_case_pct": ":.2f",
                "rank_total_cases": True,
                cat_col: True,
            },
            labels={
                cat_col: "Kategori Penyebaran",
                val_col: val_title,
                "total_cases": "Total Kasus Kumulatif",
                "cases_2024": "Kasus 2024",
                "incidence_rate": "Insidens per 100k",
                "national_case_pct": "% Nasional",
                "rank_total_cases": "Peringkat Nasional",
            },
            map_style="open-street-map",
            center={"lat": -2.5, "lon": 118.0},
            zoom=3.8,
            opacity=0.65,
        )
    else:
        fig = px.scatter_map(
            summary_df,
            lat="lat",
            lon="lon",
            color=cat_col,
            color_discrete_map=REAL_COLOR_MAP,
            category_orders={cat_col: REAL_CATEGORY_ORDER},
            size=np.maximum(10, np.sqrt(summary_df["total_cases"]) / 22),
            hover_name="name",
            hover_data={
                "region_id": True,
                val_col: True,
                "total_cases": ":,",
                "incidence_rate": ":.1f",
            },
            map_style="open-street-map",
            center={"lat": -2.5, "lon": 118.0},
            zoom=3.8,
        )

    # Add Bubble Markers on Centroids for visual pop
    if "lon" in summary_df and "lat" in summary_df:
        fig.add_trace(
            go.Scattermap(
                lon=summary_df["lon"],
                lat=summary_df["lat"],
                hovertext=summary_df["name"],
                hoverinfo="text",
                mode="markers",
                marker=dict(
                    size=np.maximum(8, np.sqrt(summary_df["total_cases"]) / 24),
                    color=summary_df[cat_col].map(REAL_COLOR_MAP),
                    opacity=0.9,
                ),
                showlegend=False,
                name="Centroid Titik",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            title_text="<b>Kategori Penyebaran Kasus:</b> ",
            font=dict(size=12),
        ),
    )
    return fig


def create_fan_chart(
    history_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    region_id: str,
) -> go.Figure:
    """Render fan chart of historical cases + probabilistic forecast cone."""
    fig = go.Figure()

    # 1. Historical line
    hist_sub = history_df[history_df["region_id"] == region_id].sort_values("month")
    if not hist_sub.empty:
        fig.add_trace(
            go.Scatter(
                x=hist_sub["month"],
                y=hist_sub["cases"],
                mode="lines+markers",
                name="Kasus Historis (Aktual)",
                line=dict(color="#1f77b4", width=2.5),
                marker=dict(size=4),
            )
        )

    # 2. Forecast points
    fc_sub = forecasts_df[forecasts_df["region_id"] == region_id].sort_values("lead")
    if not fc_sub.empty:
        # Connect last history point to first forecast
        if not hist_sub.empty:
            last_hist_month = hist_sub["month"].iloc[-1]
            last_hist_cases = hist_sub["cases"].iloc[-1]
            x_fc = [last_hist_month] + list(fc_sub["target"])
            y_q025 = [last_hist_cases] + list(fc_sub["q025"])
            y_q50 = [last_hist_cases] + list(fc_sub["q50"])
            y_q975 = [last_hist_cases] + list(fc_sub["q975"])
        else:
            x_fc = list(fc_sub["target"])
            y_q025 = list(fc_sub["q025"])
            y_q50 = list(fc_sub["q50"])
            y_q975 = list(fc_sub["q975"])

        # 95% Interval
        fig.add_trace(
            go.Scatter(
                x=x_fc + x_fc[::-1],
                y=y_q975 + y_q025[::-1],
                fill="toself",
                fillcolor="rgba(253, 174, 97, 0.25)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="Interval Kredibel 95%",
            )
        )

        # Median forecast
        fig.add_trace(
            go.Scatter(
                x=x_fc,
                y=y_q50,
                mode="lines+markers",
                name="Prakiraan Median (q50)",
                line=dict(color="#d7191c", width=3, dash="dash"),
                marker=dict(size=6, symbol="diamond"),
            )
        )

    fig.update_layout(
        title=f"Riwayat Epidemiologi & Kerucut Prakiraan — {region_id}",
        xaxis_title="Bulan (YYYY-MM)",
        yaxis_title="Jumlah Kasus Bulanan",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=420,
    )
    return fig


def create_exceedance_bar(forecast_row: pd.Series | dict[str, Any]) -> go.Figure:
    """Render horizontal bar chart of threshold exceedance probabilities."""
    labels = [
        "Rata-rata + 1 SD (Waspada)",
        "Rata-rata + 2 SD (Siaga/KLB)",
        "Persentil 75 Historis",
        "Persentil 95 Historis",
    ]
    vals = [
        float(forecast_row.get("p_exceed_m1sd", 0.0)),
        float(forecast_row.get("p_exceed_m2sd", 0.0)),
        float(forecast_row.get("p_exceed_p75", 0.0)),
        float(forecast_row.get("p_exceed_p95", 0.0)),
    ]

    colors = []
    for v in vals:
        if v >= 0.70:
            colors.append("#d7191c")
        elif v >= 0.40:
            colors.append("#fdae61")
        elif v >= 0.20:
            colors.append("#fecc5c")
        else:
            colors.append("#2b83ba")

    fig = go.Figure(
        go.Bar(
            x=vals,
            y=labels,
            orientation="h",
            marker=dict(color=colors),
            text=[f"{v*100:.1f}%" for v in vals],
            textposition="auto",
        )
    )

    fig.update_layout(
        title="Probabilitas Melampaui Ambang Epidemiologis",
        xaxis=dict(title="Probabilitas", range=[0, 1.05], tickformat=".0%"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=10, r=20, t=40, b=20),
        height=260,
    )
    return fig


def create_crpss_chart(scores_df: pd.DataFrame) -> go.Figure:
    """Render CRPSS per lead bar chart."""
    if scores_df.empty or "lead" not in scores_df.columns:
        return go.Figure()

    lead_grp = scores_df.groupby("lead")["crpss"].median().reset_index()

    colors = ["#2b83ba" if v > 0 else "#d7191c" for v in lead_grp["crpss"]]

    fig = go.Figure(
        go.Bar(
            x=[f"Lead {int(lead_val)} bln" for lead_val in lead_grp["lead"]],
            y=lead_grp["crpss"],
            marker=dict(color=colors),
            text=[f"{v:+.3f}" for v in lead_grp["crpss"]],
            textposition="auto",
        )
    )

    fig.add_hline(
        y=0.0,
        line_dash="dash",
        line_color="black",
        annotation_text="Batas Skill Baseline (CRPSS = 0)",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Skill Prakiraan (CRPSS Median vs Baseline Musiman B1)",
        xaxis_title="Lead Time (Bulan ke Depan)",
        yaxis_title="CRPSS (Skor Keterampilan)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=360,
    )
    return fig


def create_reliability_curve(rel_df: pd.DataFrame) -> go.Figure:
    """Render 10-bin reliability diagram with diagonal reference."""
    fig = go.Figure()

    # Reference diagonal
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="gray"),
            name="Kalibrasi Sempurna",
        )
    )

    if not rel_df.empty and "forecast_prob" in rel_df.columns and "observed_freq" in rel_df.columns:
        fig.add_trace(
            go.Scatter(
                x=rel_df["forecast_prob"],
                y=rel_df["observed_freq"],
                mode="lines+markers",
                name="Kurva Reliabilitas",
                line=dict(color="#2b83ba", width=2),
                marker=dict(size=7),
            )
        )

    fig.update_layout(
        title="Kurva Reliabilitas (Kalibrasi Probabilitas)",
        xaxis=dict(title="Prakiraan Probabilitas", range=[0, 1]),
        yaxis=dict(title="Frekuensi Teramati Riil", range=[0, 1]),
        margin=dict(l=40, r=40, t=50, b=40),
        height=360,
    )
    return fig


def create_weights_pie(weights_df: pd.DataFrame) -> go.Figure:
    """Render superensemble weights distribution."""
    if weights_df.empty:
        return go.Figure()

    fig = px.pie(
        weights_df,
        names="model_id",
        values="weight",
        title="Bobot Superensemble ($w_i \\propto 1/\\text{CRPS}_i^2$)",
        hole=0.4,
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=320)
    return fig
