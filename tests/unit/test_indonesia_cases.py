"""Unit tests for Indonesian cases and administrative data adapters (P3c)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Polygon

from denguecast.data.ingest.geo import compute_adjacency
from denguecast.data.ingest.opendengue import aggregate_weekly_to_monthly
from denguecast.data.ingest.opendengue import load as load_opendengue
from denguecast.data.ingest.population import incidence, interpolate_population


def test_aggregate_weekly_to_monthly() -> None:
    """Test sum of weekly cases into calendar months."""
    weekly_df = pd.DataFrame(
        {
            "region_id": ["ID-JK", "ID-JK", "ID-JK", "ID-JK"],
            "date": ["2020-01-05", "2020-01-12", "2020-01-19", "2020-02-02"],
            "dengue_cases": [10, 20, 30, 40],
        }
    )

    monthly = aggregate_weekly_to_monthly(weekly_df, date_col="date", case_col="dengue_cases")
    assert len(monthly) == 2

    jan = monthly[monthly["month"] == "2020-01"].iloc[0]
    feb = monthly[monthly["month"] == "2020-02"].iloc[0]

    assert jan["cases"] == 60
    assert feb["cases"] == 40


def test_opendengue_alias_normalization(tmp_path: Path) -> None:
    """Test that Indonesian province variations map to canonical region_id."""
    alias_file = tmp_path / "aliases.csv"
    alias_file.write_text(
        "alias,region_id,canonical_name\n"
        "DKI Jakarta,ID-JK,DKI Jakarta\n"
        "Jawa Barat,ID-JB,Jawa Barat\n",
        encoding="utf-8",
    )

    cases_file = tmp_path / "opendengue_sample.csv"
    df = pd.DataFrame(
        {
            "country": ["IDN", "IDN", "IDN"],
            "adm_1_name": ["DKI Jakarta", "Jawa Barat", "Provinsi Asing"],
            "calendar_start_date": ["2020-01-01", "2020-01-01", "2020-01-01"],
            "dengue_cases": [100, 250, 50],
            "definition": ["clinical", "clinical", "clinical"],
        }
    )
    df.to_csv(cases_file, index=False)

    loaded, audit = load_opendengue(cases_file, alias_path=alias_file)
    assert len(loaded) == 2  # Provinsi Asing dropped from canonical table
    assert set(loaded["region_id"].unique()) == {"ID-JK", "ID-JB"}
    assert "Provinsi Asing" in audit["unmapped_provinces"]


def test_population_interpolation_and_incidence() -> None:
    """Population interpolation and incidence calculation match hand math."""
    # BPS census 2010 = 1,000,000; 2020 = 1,100,000 (10k increase per year)
    census_df = pd.DataFrame(
        {
            "region_id": ["ID-JK", "ID-JK"],
            "year": [2010, 2020],
            "population": [1_000_000, 1_100_000],
        }
    )

    interp = interpolate_population(census_df, min_year=2010, max_year=2020)
    assert len(interp) == 11

    pop_2015 = interp[interp["year"] == 2015]["population"].iloc[0]
    assert pop_2015 == 1_050_000

    # Test incidence per 100,000:
    # 210 cases with 1,050,000 population -> 210 / 1,050,000 * 100,000 = 20.0
    cases_df = pd.DataFrame(
        {
            "region_id": ["ID-JK"],
            "month": ["2015-06"],
            "cases": [210],
            "definition": ["clinical"],
        }
    )

    inc_df = incidence(cases_df, interp)
    assert np.isclose(inc_df["incidence"].iloc[0], 20.0)


def test_compute_adjacency_symmetry() -> None:
    """Adjacency between spatial polygons must be symmetric."""
    import geopandas as gpd

    # Three adjacent 1x1 squares along x-axis: [0,1], [1,2], [2,3]
    p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
    p2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])
    p3 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])

    gdf = gpd.GeoDataFrame(
        {
            "region_id": ["R1", "R2", "R3"],
            "geometry": [p1, p2, p3],
        }
    )

    adj = compute_adjacency(gdf, id_col="region_id")

    assert adj["R1"] == ["R2"]
    assert adj["R2"] == ["R1", "R3"]
    assert adj["R3"] == ["R2"]

    # Verify symmetry
    for r_a, neighbors in adj.items():
        for r_b in neighbors:
            assert r_a in adj[r_b]
