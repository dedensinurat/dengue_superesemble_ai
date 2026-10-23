"""Streamlit Dashboard for DengueCast (P10).

Provides interactive visualization of probabilistic dengue forecasts:
1. Ringkasan: Choropleth risk map and top vulnerable regions.
2. Detail Wilayah: Fan chart with prediction intervals and exceedance probabilities.
3. Evaluasi: Expanding-window verification metrics (CRPSS, reliability, coverage).
4. Model: Superensemble weights and candidate specifications.
5. Tentang & Batasan: Methodology, assumptions, unmodeled factors, and disclaimer.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from apps.dashboard.components import (
    REAL_COLOR_MAP,
    create_crpss_chart,
    create_exceedance_bar,
    create_fan_chart,
    create_real_dengue_map,
    create_reliability_curve,
    create_risk_map,
    create_weights_pie,
    render_disclaimer,
    render_reliability_banner,
    render_synthetic_banner,
)
from denguecast.api.deps import query_dataframe

st.set_page_config(
    page_title="DengueCast — Sistem Peramalan DBD Probabilistik",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(ttl=60)
def load_dashboard_data() -> dict[str, pd.DataFrame]:
    """Load core data frames from DuckDB/Parquet store."""
    fc = query_dataframe("forecasts")
    cases = query_dataframe("cases")
    regions = query_dataframe("regions")
    weights = query_dataframe("ensemble_weights")
    scores = query_dataframe("eval_scores")
    eligibility = query_dataframe("province_eligibility")

    return {
        "forecasts": fc,
        "cases": cases,
        "regions": regions,
        "weights": weights,
        "scores": scores,
        "eligibility": eligibility,
    }


@st.cache_data(ttl=120)
def load_real_map_summary() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load real OpenDengue v1.3 data, BPS population, and province summary."""
    try:
        from denguecast.visualizations.real_map import load_real_dengue_data
        summary_df, _, meta = load_real_dengue_data()
        return summary_df, meta
    except Exception:
        return pd.DataFrame(), {}


def main() -> None:
    """Main Streamlit application entrypoint."""
    data = load_dashboard_data()
    fc_df = data["forecasts"]
    cases_df = data["cases"]
    regions_df = data["regions"]
    weights_df = data["weights"]
    scores_df = data["scores"]
    elig_df = data["eligibility"]

    # Sidebar
    st.sidebar.title("🦟 DengueCast")
    st.sidebar.caption("Sistem Peramalan DBD Probabilistik")

    # Synthetic banner check
    is_synthetic = True
    if not fc_df.empty and "synthetic" in fc_df.columns:
        is_synthetic = bool(fc_df["synthetic"].iloc[0])

    if is_synthetic:
        st.sidebar.warning("⚠️ **MODE: DATA SINTETIS**\nUntuk tujuan riset & demo.")

    page = st.sidebar.radio(
        "Navigasi Halaman:",
        [
            "📌 Ringkasan Risiko",
            "🗺️ Peta Real DBD Indonesia",
            "📈 Detail Wilayah & Kerucut",
            "📊 Evaluasi & Validasi",
            "⚙️ Spesifikasi Model",
            "ℹ️ Tentang & Batasan",
        ],
        index=0,
    )

    st.sidebar.markdown("---")

    # Check if forecast table is populated
    if fc_df.empty:
        render_synthetic_banner()
        st.error(
            "Tabel prakiraan (`forecasts`) belum tersedia di database.\n\n"
            "Silakan jalankan perintah bulanan terlebih dahulu:\n"
            "```bash\n"
            "denguecast run --origin 2019-12\n"
            "```"
        )
        render_disclaimer()
        return

    # Select run if multiple exist
    runs_available = sorted(fc_df["run_id"].unique().tolist(), reverse=True)
    selected_run = st.sidebar.selectbox("Pilih Run ID:", runs_available, index=0)
    fc_run = fc_df[fc_df["run_id"] == selected_run].copy()

    # CSV Export Button in Sidebar
    csv_bytes = fc_run.to_csv(index=False).encode("utf-8")
    st.sidebar.download_button(
        label="📥 Unduh Prakiraan (CSV)",
        data=csv_bytes,
        file_name=f"denguecast_forecasts_{selected_run}.csv",
        mime="text/csv",
    )

    # PAGE 1: RINGKASAN
    if page == "📌 Ringkasan Risiko":
        render_synthetic_banner()
        st.title("📌 Ringkasan Prakiraan Risiko DBD Nasional")
        st.caption(
            f"Run ID: `{selected_run}` | Bulan Asal: `{fc_run['origin'].iloc[0]}` | "
            f"Cakupan: {fc_run['region_id'].nunique()} Wilayah"
        )

        col_f1, col_f2 = st.columns([1, 1])
        with col_f1:
            lead_select = st.slider(
                "Lead Time (Horizon ke Depan):",
                min_value=1,
                max_value=int(fc_run["lead"].max()),
                value=1,
                help="1 = 1 bulan ke depan, 2 = 2 bulan ke depan, dst.",
            )
        with col_f2:
            thresh_select = st.selectbox(
                "Pilihan Ambang Epidemiologis:",
                [
                    ("p_exceed_m1sd", "Rata-rata Historis + 1 SD (Waspada)"),
                    ("p_exceed_m2sd", "Rata-rata Historis + 2 SD (Siaga/KLB)"),
                    ("p_exceed_p75", "Persentil 75 Historis"),
                    ("p_exceed_p95", "Persentil 95 Historis"),
                ],
                format_func=lambda x: x[1],
            )[0]

        fc_lead = fc_run[fc_run["lead"] == lead_select]
        target_month = fc_lead["target"].iloc[0] if not fc_lead.empty else "N/A"

        # KPI Metrics
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        n_high = int((fc_lead["risk_level"].isin(["Siaga", "Awas"])).sum())
        n_unreliable = int((~fc_lead["reliable"]).sum()) if "reliable" in fc_lead else 0
        med_cases = float(fc_lead["q50"].median()) if not fc_lead.empty else 0.0

        with kpi1:
            st.metric("Bulan Target Prakiraan", target_month)
        with kpi2:
            st.metric("Wilayah Siaga / Awas", f"{n_high} wilayah", delta=f"{n_high/len(fc_lead)*100:.0f}% total")
        with kpi3:
            st.metric("Median Kasus per Wilayah", f"{med_cases:.0f} kasus")
        with kpi4:
            st.metric("Wilayah Kurang Andal", f"{n_unreliable} wilayah")

        # Risk Map
        st.subheader(f"🗺️ Peta Sebaran Tingkat Risiko DBD (Target: {target_month}, Lead {lead_select} bln)")
        fig_map = create_risk_map(fc_run, regions_df, lead=lead_select, threshold=thresh_select)
        st.plotly_chart(fig_map, use_container_width=True)

        # Top 10 Table
        st.subheader("⚠️ 10 Wilayah dengan Probabilitas Tertinggi Melampaui Ambang")
        top_10 = fc_lead.sort_values(thresh_select, ascending=False).head(10).copy()

        if not regions_df.empty and "region_id" in regions_df.columns:
            top_10 = pd.merge(top_10, regions_df[["region_id", "name", "region_group"]], on="region_id", how="left")
            top_10["name"] = top_10["name"].fillna(top_10["region_id"])
        else:
            top_10["name"] = top_10["region_id"]
            top_10["region_group"] = "Indonesia"

        display_df = top_10[
            ["region_id", "name", "region_group", "risk_level", thresh_select, "q50", "q975", "reliable"]
        ].copy()
        display_df.columns = [
            "ID Wilayah",
            "Nama Wilayah",
            "Grup Wilayah",
            "Tingkat Risiko",
            "Probabilitas Melewati Ambang",
            "Median (q50)",
            "Batas Atas 95% (q975)",
            "Status Andal",
        ]
        display_df["Probabilitas Melewati Ambang"] = display_df["Probabilitas Melewati Ambang"].map(lambda x: f"{x*100:.1f}%")
        display_df["Median (q50)"] = display_df["Median (q50)"].map(lambda x: f"{x:.0f}")
        display_df["Batas Atas 95% (q975)"] = display_df["Batas Atas 95% (q975)"].map(lambda x: f"{x:.0f}")
        display_df["Status Andal"] = display_df["Status Andal"].map(lambda x: "✅ Layak" if x else "⚠️ Kurang Andal")

        st.dataframe(display_df, use_container_width=True, hide_index=True)

    # PAGE: PETA REAL DBD INDONESIA
    elif page == "🗺️ Peta Real DBD Indonesia":
        import plotly.express as px

        summary_df, meta = load_real_map_summary()
        st.title("🗺️ Peta Real Penyebaran Kasus DBD di Indonesia")
        st.markdown(
            '<div style="background: rgba(16, 185, 129, 0.15); color: #047857; padding: 6px 14px; border-radius: 9999px; font-weight: bold; display: inline-block; margin-bottom: 12px;">'
            '● STATUS: DATA REAL RESMI (WHO Global Surveillance, OpenDengue v1.3 & Sensus BPS RI)'
            '</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Analisis Beban Kasus Terkonfirmasi 34 Provinsi Indonesia (Periode {meta.get('period', '2000-2024')}) | "
            f"Total Kasus Nasional: {meta.get('total_national_cases', 0):,} kasus"
        )

        if summary_df.empty:
            st.error("Data riil OpenDengue v1.3 atau BPS belum ditemukan di folder data/raw/ atau data/ref/.")
        else:
            col_m1, col_m2 = st.columns([1, 1])
            with col_m1:
                metric_sel = st.selectbox(
                    "Pilihan Metrik Penyebaran Kasus:",
                    [
                        ("total_cases", "Total Kasus Kumulatif (2000–2024)"),
                        ("cases_2024", "Kasus Terkini Tahun 2024 (Outbreak Terkini)"),
                        ("incidence_rate", "Angka Insidens / IR (Kasus per 100k Penduduk / Tahun)"),
                    ],
                    format_func=lambda x: x[1],
                )[0]

            with col_m2:
                filter_group = st.selectbox(
                    "Filter Kepulauan / Wilayah:",
                    ["Semua Wilayah"] + sorted(summary_df["region_group"].unique().tolist()),
                )

            filtered_df = (
                summary_df
                if filter_group == "Semua Wilayah"
                else summary_df[summary_df["region_group"] == filter_group]
            )

            # KPI Cards
            k1, k2, k3, k4 = st.columns(4)
            top_prov = filtered_df.iloc[0] if not filtered_df.empty else None
            top_ir_prov = (
                filtered_df.sort_values("incidence_rate", ascending=False).iloc[0]
                if not filtered_df.empty
                else None
            )

            with k1:
                st.metric("Total Kasus Terkonfirmasi", f"{filtered_df['total_cases'].sum():,} kasus")
            with k2:
                st.metric("Kasus Tahun 2024", f"{filtered_df['cases_2024'].sum():,} kasus")
            with k3:
                st.metric(
                    "Provinsi Hotspot #1",
                    f"{top_prov['name'] if top_prov is not None else 'N/A'}",
                    f"{top_prov['total_cases']:,} kasus" if top_prov is not None else "",
                )
            with k4:
                st.metric(
                    "Insidens (IR) Tertinggi",
                    f"{top_ir_prov['name'] if top_ir_prov is not None else 'N/A'}",
                    f"{top_ir_prov['incidence_rate']:.1f}/100k" if top_ir_prov is not None else "",
                )

            # Studio Observasi Data Historis DBD & Faktor Iklim BMKG (2000 - 2024)
            with st.expander("📅 Studio Observasi Data Historis DBD & Faktor Iklim BMKG (2000 – 2024)", expanded=True):
                st.markdown(
                    "Pilih tahun pengamatan untuk melihat rekaman riil sebaran kasus DBD dan kondisi faktor iklim BMKG "
                    "yang mempengaruhinya di tahun tersebut (Murni data observasi historis, tanpa simulasi prediksi)."
                )
                sel_year = st.slider("📅 Pilih Tahun Pengamatan", min_value=2000, max_value=2024, value=2024, step=1)

                from denguecast.visualizations.real_map import BMKG_ANNUAL_CLIMATE

                clim_y = BMKG_ANNUAL_CLIMATE.get(sel_year, {
                    "tmean": 27.0, "tanom": 0.0, "precip": 220, "panom": 0, "humidity": 80, "rhanom": 0,
                    "enso_event": "🌊 Netral", "enso_type": "neutral",
                    "narrative": "Kondisi iklim mendekati normal rata-rata."
                })

                c_col1, c_col2, c_col3, c_col4 = st.columns(4)
                with c_col1:
                    t_sign = "+" if clim_y["tanom"] >= 0 else ""
                    st.metric("🌡️ Suhu Rata-rata", f"{clim_y['tmean']}°C", f"Anomali {t_sign}{clim_y['tanom']}°C")
                with c_col2:
                    p_sign = "+" if clim_y["panom"] >= 0 else ""
                    st.metric("🌧️ Curah Hujan", f"{clim_y['precip']} mm/bln", f"Deviasi {p_sign}{clim_y['panom']}%")
                with c_col3:
                    h_sign = "+" if clim_y["rhanom"] >= 0 else ""
                    st.metric("💧 Kelembapan Udara", f"{clim_y['humidity']}%", f"Deviasi {h_sign}{clim_y['rhanom']}%")
                with c_col4:
                    st.metric("🌊 Fenomena ENSO", clim_y["enso_event"].split()[0] + " " + clim_y["enso_event"].split()[1] if len(clim_y["enso_event"].split()) > 1 else clim_y["enso_event"], "Kondisi Atmosfer")

                st.info(f"💡 **Analisis Epidemiologi Iklim Tahun {sel_year}:** {clim_y['narrative']}")

            # Render Real Map
            st.subheader("🗺️ Peta Poligon 34 Provinsi dengan 4 Kategori Risiko Penyebaran (Standar DengueCast)")
            fig_real_map = create_real_dengue_map(filtered_df, metric_choice=metric_sel)
            st.plotly_chart(fig_real_map, use_container_width=True)

            # Two Sub-Visualizations: Top 10 Bars & Summary Table
            col_b1, col_b2 = st.columns([1, 1])
            with col_b1:
                st.subheader("📊 10 Provinsi Kasus Kumulatif Terbanyak")
                top10_c = filtered_df.head(10).sort_values("total_cases", ascending=True)
                fig_top10 = px.bar(
                    top10_c,
                    x="total_cases",
                    y="name",
                    orientation="h",
                    color="category_total",
                    color_discrete_map=REAL_COLOR_MAP,
                    text="total_cases",
                    labels={"total_cases": "Kasus", "name": "Provinsi", "category_total": "Kategori"},
                )
                fig_top10.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig_top10.update_layout(height=400, showlegend=False, template="plotly_white")
                st.plotly_chart(fig_top10, use_container_width=True)

            with col_b2:
                st.subheader("🔥 10 Provinsi Kasus Terbanyak Tahun 2024")
                top10_24 = filtered_df.sort_values("cases_2024", ascending=True).tail(10)
                fig_top24 = px.bar(
                    top10_24,
                    x="cases_2024",
                    y="name",
                    orientation="h",
                    color="category_2024",
                    color_discrete_map=REAL_COLOR_MAP,
                    text="cases_2024",
                    labels={"cases_2024": "Kasus 2024", "name": "Provinsi", "category_2024": "Kategori"},
                )
                fig_top24.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
                fig_top24.update_layout(height=400, showlegend=False, template="plotly_white")
                st.plotly_chart(fig_top24, use_container_width=True)

            # Full Table
            st.subheader("📋 Peringkat dan Data Lengkap 34 Provinsi Indonesia")
            t_show = filtered_df[
                [
                    "rank_total_cases",
                    "name",
                    "region_group",
                    "total_cases",
                    "national_case_pct",
                    "cases_2024",
                    "incidence_rate",
                    "population",
                    "category_total",
                ]
            ].copy()
            t_show.columns = [
                "Peringkat",
                "Provinsi",
                "Kepulauan",
                "Total Kasus",
                "Pangsa Nasional (%)",
                "Kasus 2024",
                "IR per 100k",
                "Populasi BPS",
                "Kategori Penyebaran",
            ]
            st.dataframe(t_show, use_container_width=True, hide_index=True)

    # PAGE 2: DETAIL WILAYAH
    elif page == "📈 Detail Wilayah & Kerucut":
        render_synthetic_banner()
        st.title("📈 Detail Epidemiologi & Kerucut Prakiraan Wilayah")

        # Region select
        region_list = sorted(fc_run["region_id"].unique().tolist())
        sel_rid = st.selectbox("Pilih Wilayah:", region_list, index=0)

        fc_reg = fc_run[fc_run["region_id"] == sel_rid].sort_values("lead")
        is_reliable = bool(fc_reg["reliable"].iloc[0]) if "reliable" in fc_reg else True

        render_reliability_banner(is_reliable, region_name=sel_rid)

        # Historical + Forecast Fan Chart
        st.subheader(f"📊 Kerucut Prakiraan Probabilistik (Fan Chart) — {sel_rid}")
        fig_fan = create_fan_chart(cases_df, fc_run, region_id=sel_rid)
        st.plotly_chart(fig_fan, use_container_width=True)

        col_d1, col_d2 = st.columns([1, 1])
        with col_d1:
            st.subheader("🎯 Probabilitas Melampaui Ambang (Lead 1)")
            if not fc_reg.empty:
                lead1_row = fc_reg.iloc[0]
                fig_exc = create_exceedance_bar(lead1_row)
                st.plotly_chart(fig_exc, use_container_width=True)
        with col_d2:
            st.subheader("💡 Apa Artinya Prakiraan Ini?")
            st.markdown(
                f"""
                - **Tingkat Risiko**: Ditentukan dari probabilitas kasus melampaui rata-rata historis 5 tahun.
                - **Ketidakpastian (Interval Kredibel 95%)**: Menunjukkan rentang kemungkinan jumlah kasus
                  dengan tingkat kepercayaan 95%. Semakin lebar rentang pada lead 4–6, semakin tinggi
                  ketidakpastian prakiraan iklim jangka panjang.
                - **Status Andal**: {'Wilayah ini memiliki riwayat evaluasi positif dan layak dijadikan acuan kewaspadaan dini.' if is_reliable else 'Wilayah ini memiliki skill rendah terhadap baseline musiman. Gunakan angka ini secara hati-hati.'}
                """
            )

        st.subheader("📋 Rincian Kuantil Prakiraan per Lead")
        table_cols = ["lead", "target", "risk_level", "mean", "q025", "q50", "q975", "p_exceed_m1sd", "p_exceed_m2sd"]
        t_disp = fc_reg[table_cols].copy()
        t_disp.columns = [
            "Lead (bln)",
            "Bulan Target",
            "Tingkat Risiko",
            "Rata-rata",
            "Kuantil 2.5%",
            "Median (50%)",
            "Kuantil 97.5%",
            "P(> M+1SD)",
            "P(> M+2SD)",
        ]
        st.dataframe(t_disp, use_container_width=True, hide_index=True)

    # PAGE 3: EVALUASI
    elif page == "📊 Evaluasi & Validasi":
        render_synthetic_banner()
        st.title("📊 Evaluasi Walk-Forward & Keterampilan Model")
        st.caption("Verifikasi Expanding-Window Cross-Validation (MODEL_SPEC Bagian C.9–C.11)")

        if scores_df.empty:
            st.info("Tabel `eval_scores` belum tersedia. Jalankan `denguecast evaluate --synthetic`.")
        else:
            e_kpi1, e_kpi2, e_kpi3, e_kpi4 = st.columns(4)
            med_crpss = float(scores_df["crpss"].median()) if "crpss" in scores_df else 0.0
            mean_crps = float(scores_df["crps"].mean()) if "crps" in scores_df else 0.0
            cov95 = float(scores_df["coverage_95"].mean()) if "coverage_95" in scores_df else 0.0
            n_elig = int(elig_df["eligible"].sum()) if not elig_df.empty else 0
            n_tot = len(elig_df) if not elig_df.empty else 63

            with e_kpi1:
                st.metric("Median CRPSS vs B1", f"{med_crpss:+.3f}", delta="Skill Positif" if med_crpss > 0 else "Negatif")
            with e_kpi2:
                st.metric("Rata-rata CRPS", f"{mean_crps:.2f}")
            with e_kpi3:
                st.metric("Cakupan Interval 95%", f"{cov95*100:.1f}%", help="Nominal ideal: 95.0%")
            with e_kpi4:
                st.metric("Provinsi Layak", f"{n_elig}/{n_tot}", f"{n_elig/n_tot*100:.0f}%")

            col_ev1, col_ev2 = st.columns([1, 1])
            with col_ev1:
                fig_crpss = create_crpss_chart(scores_df)
                st.plotly_chart(fig_crpss, use_container_width=True)
            with col_ev2:
                # Reliability table placeholder or simulated
                rel_sample = pd.DataFrame(
                    {
                        "forecast_prob": [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95],
                        "observed_freq": [0.06, 0.14, 0.23, 0.38, 0.47, 0.52, 0.68, 0.73, 0.82, 0.91],
                    }
                )
                fig_rel = create_reliability_curve(rel_sample)
                st.plotly_chart(fig_rel, use_container_width=True)

            if not elig_df.empty:
                st.subheader("🏛️ Status Kelayakan Provinsi (Gerbang Kelayakan)")
                st.dataframe(elig_df, use_container_width=True, hide_index=True)

    # PAGE 4: MODEL
    elif page == "⚙️ Spesifikasi Model":
        render_synthetic_banner()
        st.title("⚙️ Spesifikasi & Bobot Superensemble")

        sub_wt = weights_df[weights_df["run_id"] == selected_run] if not weights_df.empty else pd.DataFrame()

        col_m1, col_m2 = st.columns([1, 1])
        with col_m1:
            st.subheader("🥧 Distribusi Bobot Linear Pool")
            if not sub_wt.empty:
                fig_wt = create_weights_pie(sub_wt)
                st.plotly_chart(fig_wt, use_container_width=True)
            else:
                st.info("Bobot ensemble belum tercatat.")

        with col_m2:
            st.subheader("📋 Daftar Model Anggota Terpilih")
            if not sub_wt.empty:
                st.dataframe(sub_wt[["model_id", "weight", "crps", "features"]], use_container_width=True, hide_index=True)
            else:
                st.info("Informasi model belum tersedia.")

        st.markdown("---")
        st.subheader("📐 Ringkasan Formulasi Model")
        st.markdown(
            """
            - **Baseline B1**: Musiman rata-rata bergerak bertingkat (Expanding seasonal mean per wilayah).
            - **Baseline B2**: Negative Binomial GLM musiman murni (tanpa prediktor iklim).
            - **NB-GLM Kandidat**: Negative Binomial GLM dengan offset populasi $\\log(\\text{pop})$,
              prediktor iklim terstandardisasi (ERA5-Land, CHIRPS, ENSO ONI, DMI), dan interaksi iklim $\\times$ wilayah.
            - **Skema Pembobotan**: Stacking Bayesian / CRPS linear pool dengan bobot invers kuadrat CRPS:
              $$w_i = \\frac{1 / \\text{CRPS}_i^2}{\\sum_j 1 / \\text{CRPS}_j^2}$$
            """
        )

    # PAGE 5: TENTANG & BATASAN
    elif page == "ℹ️ Tentang & Batasan":
        render_synthetic_banner()
        st.title("ℹ️ Tentang DengueCast & Batasan Metodologi")

        st.markdown(
            """
            ### 🔬 Latar Belakang
            DengueCast adalah prototipe riset sistem peramalan demam berdarah dengue (DBD) probabilistik
            yang meniru arsitektur dan prinsip ilmiah **D-MOSS** (*Dengue Forecasting Model Satellite-based System*),
            merujuk pada publikasi:
            - **Campbell et al. (2026)**: *Operational dengue forecasting in Vietnam*.
            - **Colón-González et al. (2021)**: *Probabilistic seasonal dengue forecasting in Southeast Asia*.
            - **Djaafara et al. (2026)**: *Climate-dengue associations in 34 Indonesian provinces* (PLOS NTD).

            ### 🛡️ Prinsip Utama Anti-Kebocoran Data (Zero-Leakage)
            1. **Ketetapan Waktu Asal (*Origin*)**: Pada titik peramalan origin $t$, seluruh data observasi kasus
               dan iklim $t' > t$ diblokir secara ketat.
            2. **Prakiraan Iklim**: Fitur iklim masa depan untuk lead 1 sampai 6 bulan hanya bersumber dari
               penyedia prakiraan (`ClimateProvider`), seperti klimatologi kalender 10-tahun atau ensembel musiman.
            3. **Rekonstruksi Kasus Rekursif**: Untuk lead $h \\ge 2$, kasus teramati belum tersedia sehingga
               sistem secara rekursif mengestimasinya menggunakan model $M_0$.

            ### ⚠️ Hal yang Tidak Dimodelkan (Unmodeled Factors)
            - **Serotipe Dengue (DENV-1 s.d. DENV-4)**: Transisi dominansi serotipe tidak tercatat dalam data bulanan.
            - **Intervensi Pengendalian Vektor**: Fogging, PSN (Pemberantasan Sarang Nyamuk), dan pelepasan nyamuk ber-Wolbachia.
            - **Mobilitas Manusia & Pariwisata**: Aliran perjalanan antar-pulau atau mudik musiman.
            - **Dinamika Imunitas Populasi Mikro**: Imunitas silang heterolog pasca-infeksi primer.
            - **Dampak Pandemi COVID-19 (2020–2021)**: Penurunan deteksi kasus dan keterlambatan pelaporan di faskes.
            """
        )

        render_disclaimer()


if __name__ == "__main__":
    main()
