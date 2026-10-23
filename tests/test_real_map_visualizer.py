"""Unit tests for Real Dengue Distribution Map Visualizer (real_map.py)."""

from __future__ import annotations

from pathlib import Path

from denguecast.visualizations.real_map import (
    CANONICAL_NATIONAL_METRICS,
    CATEGORY_ORDER,
    COLOR_MAP,
    DengueMapConfig,
    categorize_value,
    generate_high_res_static_map,
    generate_interactive_html_map,
    load_real_dengue_data,
)


def test_color_map_and_categories_consistent() -> None:
    """Ensure color map covers all defined categories."""
    assert len(COLOR_MAP) == 4
    assert len(CATEGORY_ORDER) == 4
    assert set(CATEGORY_ORDER) == {"Awas", "Siaga", "Waspada", "Rendah"}
    for cat in CATEGORY_ORDER:
        assert cat in COLOR_MAP
        assert COLOR_MAP[cat].startswith("#")


def test_categorize_value() -> None:
    """Test value categorization against cutoffs."""
    cutoffs = [150000, 70000, 25000]
    assert categorize_value(500000, cutoffs) == "Awas"
    assert categorize_value(150000, cutoffs) == "Awas"
    assert categorize_value(100000, cutoffs) == "Siaga"
    assert categorize_value(70000, cutoffs) == "Siaga"
    assert categorize_value(30000, cutoffs) == "Waspada"
    assert categorize_value(10000, cutoffs) == "Rendah"


def test_load_real_dengue_data() -> None:
    """Test loading and processing real OpenDengue data and GeoJSON."""
    cfg = DengueMapConfig()
    summary_df, geo_df, meta = load_real_dengue_data(cfg)

    # Check 34 provinces
    assert len(summary_df) == 34
    assert len(geo_df) == 34

    # Check required columns
    expected_cols = [
        "region_id",
        "name",
        "region_group",
        "total_cases",
        "cases_2024",
        "population",
        "mean_annual_cases",
        "incidence_rate",
        "national_case_pct",
        "rank_total_cases",
        "category_total",
        "category_2024",
        "category_ir",
    ]
    for col in expected_cols:
        assert col in summary_df.columns, f"Missing column: {col}"

    # Check cumulative numbers (real deduplicated total is ~2.5M cases)
    assert summary_df["total_cases"].sum() > 2000000
    assert summary_df["cases_2024"].sum() > 100000
    assert meta["total_national_cases"] > 2000000

    # Top province should be Jawa Barat
    top_prov = summary_df.iloc[0]
    assert top_prov["name"] == "Jawa Barat"
    assert top_prov["rank_total_cases"] == 1
    assert top_prov["category_total"] == "Awas"

    # Check categories subset
    assert set(summary_df["category_total"].unique()).issubset({"Awas", "Siaga", "Waspada", "Rendah"})

    # Check rankings
    assert summary_df["rank_total_cases"].min() == 1
    assert summary_df["rank_total_cases"].max() == 34

    # Check GeoDataFrame
    assert "geometry" in geo_df.columns
    assert all(geo_df["geometry"].notnull())


def test_generate_visualizations(tmp_path: Path) -> None:
    """Test generating both interactive HTML map and static PNG map."""
    cfg = DengueMapConfig()
    summary_df, geo_df, _ = load_real_dengue_data(cfg)

    html_file = tmp_path / "test_map.html"
    png_file = tmp_path / "test_map.png"

    # Test HTML generation
    out_html = generate_interactive_html_map(
        summary_df,
        geo_df,
        output_path=html_file,
        geojson_path=cfg.geojson_path,
    )
    assert out_html.exists()
    assert out_html.stat().st_size > 50000  # HTML with embedded data
    html_content = out_html.read_text(encoding="utf-8")
    assert "Peta Real Penyebaran Kasus DBD di Indonesia" in html_content
    assert "Jawa Barat" in html_content
    assert "Awas" in html_content
    assert "global-scope-bar" in html_content
    assert "country-scope-select" in html_content
    assert "global-view-container" in html_content
    assert "indonesia-view-container" in html_content
    assert "Brazil" in html_content
    assert "Singapore" in html_content

    # Test PNG generation
    out_png = generate_high_res_static_map(
        summary_df,
        geo_df,
        output_path=png_file,
    )
    assert out_png.exists()
    assert out_png.stat().st_size > 50000  # High-res PNG


def test_who_data_integrity() -> None:
    """Test WHO Dengue dataset presence and metric synchronization."""
    who_file = Path("data/raw/who_dengue_indonesia.csv")
    assert who_file.exists(), "who_dengue_indonesia.csv must exist"

    import pandas as pd
    df = pd.read_csv(who_file)
    assert len(df) >= 200, "WHO dataset should have > 200 monthly records"
    assert "CASES" in df.columns
    assert "DEATHS" in df.columns
    assert "YEAR" in df.columns

    # Verify 2024 validated WHO records
    df_2024 = df[df["YEAR"] == 2024]
    assert len(df_2024) == 12, "2024 must have 12 complete monthly records"
    cases_2024 = int(df_2024["CASES"].sum())
    deaths_2024 = int(df_2024["DEATHS"].sum())

    assert cases_2024 == 257271, f"Expected 257,271 cases in 2024, got {cases_2024}"
    assert deaths_2024 == 1461, f"Expected 1,461 deaths in 2024, got {deaths_2024}"

    # Verify canonical metrics synchronization
    assert CANONICAL_NATIONAL_METRICS[2024]["total_cases"] == 257271
    assert CANONICAL_NATIONAL_METRICS[2024]["incidence_rate"] == 92.1
    assert "WHO" in CANONICAL_NATIONAL_METRICS[2024]["milestone"]


def test_who_global_summary_and_selector() -> None:
    """Test WHO global surveillance summary structure and top country rankings."""
    import json
    summary_path = Path("data/raw/who_global_summary.json")
    assert summary_path.exists(), "who_global_summary.json must exist"

    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "countries" in data
    assert data["total_countries"] >= 100

    # Top country by 2024 cases must be Brazil
    top_c = data["countries"][0]
    assert top_c["country"] == "Brazil"
    assert top_c["global_rank_2024"] == 1
    assert top_c["cases_2024"] > 5000000

    # Singapore and Indonesia must exist
    c_names = {c["country"] for c in data["countries"]}
    assert "Singapore" in c_names
    assert "Indonesia" in c_names
    assert "Malaysia" in c_names
    assert "Viet Nam" in c_names
    assert "Thailand" in c_names

