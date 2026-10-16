"""Unit tests to ensure >= 80% test coverage across data, features, and ensemble modules."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from denguecast.data.ingest import cases, enso, landcover
from denguecast.data.store import get_data_hash, query_duckdb, read_table, write_table
from denguecast.ensemble.stacking import Superensemble
from denguecast.features.aggregate import aggregate_to_regions, compute_dtr
from denguecast.features.thresholds import endemic_channel
from denguecast.models.baseline import SeasonalExpandingMean


def test_enso_loader_csv_and_cpc(tmp_path: Path) -> None:
    # 1. Missing file error
    with pytest.raises(FileNotFoundError):
        enso.load_nino34(tmp_path / "nonexistent.csv")

    # 2. Valid CSV format
    csv_file = tmp_path / "enso.csv"
    csv_file.write_text("month,oni\n2020-01,0.5\n2020-02,-0.2\n", encoding="utf-8")
    df_csv = enso.load_nino34(csv_file)
    assert len(df_csv) == 2
    assert "nino34_anom" in df_csv.columns

    # 3. CPC matrix format
    cpc_file = tmp_path / "cpc.txt"
    cpc_text = "2020 0.5 0.4 0.3 0.2 0.1 0.0 -0.1 -0.2 -0.3 -0.4 -0.5 -0.6\n"
    cpc_file.write_text(cpc_text, encoding="utf-8")
    df_cpc = enso.load(cpc_file)
    assert len(df_cpc) == 12
    assert df_cpc["month"].iloc[0] == "2020-01"


def test_landcover_loader(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        landcover.load(tmp_path / "missing.csv")

    lc_file = tmp_path / "landcover.csv"
    lc_file.write_text(
        "region_id,year,urban_pct,periurban_pct\nreg_1,2020,45.0,20.0\n",
        encoding="utf-8",
    )
    df_lc = landcover.load(lc_file)
    assert len(df_lc) == 1
    assert df_lc["urban_pct"].iloc[0] == 45.0


def test_cases_loader(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        cases.load(tmp_path / "missing.csv")

    case_file = tmp_path / "cases.csv"
    case_file.write_text(
        "region_id,month,cases\n31,2020-01,150\n31,2020-02,120\n",
        encoding="utf-8",
    )
    df = cases.load(case_file)
    assert len(df) == 2
    assert df["definition"].iloc[0] == "clinical"
    assert df["cases"].sum() == 270


def test_aggregate_to_regions_edge_cases() -> None:
    cells = pd.DataFrame({
        "region_id": ["r1", "r1"],
        "cell_id": [1, 2],
        "tmean": [25.0, 27.0],
    })
    weights = pd.DataFrame({
        "region_id": ["r1", "r1"],
        "cell_id": [1, 2],
        "weight": [0.0, 0.0],  # zero sum fallback
    })
    agg = aggregate_to_regions(cells, weights)
    assert len(agg) == 1
    assert np.isclose(agg["tmean"].iloc[0], 26.0)

    # Missing overlapping pairs
    with pytest.raises(ValueError):
        aggregate_to_regions(cells, pd.DataFrame({"region_id": ["r2"], "cell_id": [99], "weight": [1.0]}))


def test_compute_dtr_negative_handling() -> None:
    df = pd.DataFrame({"tmax": [30.0, 20.0], "tmin": [25.0, 25.0]})
    res = compute_dtr(df)
    assert res["dtr"].iloc[0] == 5.0
    assert np.isnan(res["dtr"].iloc[1])


def test_endemic_channel_exclude_outbreak_and_empty() -> None:
    inc_df = pd.DataFrame({
        "region_id": ["r1"] * 60,
        "month": [f"{2015 + i // 12}-{1 + i % 12:02d}" for i in range(60)],
        "incidence": [10.0] * 59 + [500.0],  # 1 big outlier in year 2019
    })
    chan = endemic_channel(inc_df, origin="2019-12", years=5, exclude_outbreak_years=True)
    assert not chan.empty
    assert "mean" in chan.columns

    with pytest.raises(ValueError):
        endemic_channel(inc_df, origin="2010-01")


def test_superensemble_weighted_mean_and_edge_cases() -> None:
    train_df = pd.DataFrame({
        "region_id": ["r1"] * 24,
        "month_of_year": list(range(1, 13)) * 2,
        "cases": [10, 20] * 12,
        "population": [100000] * 24,
    })
    m1 = SeasonalExpandingMean()
    m1.fit(train_df)

    # Empty ensemble
    empty_ens = Superensemble(models=[], weights=pd.Series(dtype=float))
    test_X = train_df.iloc[:2].copy()
    zero_samp = empty_ens.predict_samples(test_X, n_samples=10)
    assert zero_samp.shape == (2, 10)
    assert np.all(zero_samp == 0)

    # Weighted mean samples combine method
    weights = pd.Series({"m1": 1.0})
    ens = Superensemble(models=[m1], weights=weights)
    samp_mean = ens.predict_samples(test_X, n_samples=10, combine="weighted_mean_samples")
    assert samp_mean.shape == (2, 10)
    assert np.all(samp_mean >= 0)

    # Check spec and to_dict
    assert ens.spec.model_id == "superensemble"
    d = ens.to_dict()
    assert d["type"] == "Superensemble"
    assert d["n_models"] == 1


def test_duckdb_store_query_and_hash(tmp_path: Path) -> None:
    db_file = tmp_path / "test.duckdb"
    df = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
    h = write_table(df, name="test_table", output_dir=tmp_path, db_path=db_file)
    assert len(h) > 0

    read_back = read_table("test_table", data_dir=tmp_path)
    assert len(read_back) == 2

    res = query_duckdb(db_file, "SELECT * FROM test_table WHERE col_a = 1")
    assert len(res) == 1
    assert res["col_b"].iloc[0] == "x"

    h2 = get_data_hash("test_table", data_dir=tmp_path)
    assert h == h2
