"""Unit tests for BMKG map image digitizer and PDF extractor (P3d)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

from denguecast.data.ingest.bmkg_image import classify_pixels, digitize, fit_affine_georef
from denguecast.data.ingest.bmkg_pdf import extract_table


def test_affine_georeference_fit_and_rejection() -> None:
    """Affine fit must calculate low RMSE on linear points and reject noisy points."""
    # Perfect linear relationship: lat = -0.1 * y, lon = 0.1 * x + 95.0
    good_points = [
        {"pixel_x": 0.0, "pixel_y": 0.0, "lat": 0.0, "lon": 95.0},
        {"pixel_x": 100.0, "pixel_y": 0.0, "lat": 0.0, "lon": 105.0},
        {"pixel_x": 0.0, "pixel_y": 100.0, "lat": -10.0, "lon": 95.0},
        {"pixel_x": 100.0, "pixel_y": 100.0, "lat": -10.0, "lon": 105.0},
    ]

    lat_c, lon_c, rmse = fit_affine_georef(good_points, max_rmse_km=50.0)
    assert rmse < 1.0  # Perfect fit

    # Corrupted points exceeding 50 km threshold
    bad_points = [
        {"pixel_x": 0.0, "pixel_y": 0.0, "lat": 0.0, "lon": 95.0},
        {"pixel_x": 100.0, "pixel_y": 0.0, "lat": 15.0, "lon": 105.0},  # Severely skewed
        {"pixel_x": 0.0, "pixel_y": 100.0, "lat": -10.0, "lon": 95.0},
        {"pixel_x": 100.0, "pixel_y": 100.0, "lat": -25.0, "lon": 120.0},
    ]

    with pytest.raises(ValueError, match="exceeds maximum allowable threshold"):
        fit_affine_georef(bad_points, max_rmse_km=50.0)


def test_classify_pixels_and_ignore_colors() -> None:
    """Pixels must match closest legend class and ignore_colors become -1."""
    legend = [
        {"label": "Low", "rgb": [255, 255, 0], "lower": 0.0, "upper": 100.0},  # Yellow
        {"label": "Medium", "rgb": [0, 255, 0], "lower": 100.0, "upper": 200.0},  # Green
        {"label": "High", "rgb": [0, 0, 255], "lower": 200.0, "upper": 300.0},  # Blue
    ]
    ignore_colors = [[255, 255, 255]]  # White ocean / background

    # 2x2 image:
    # (0,0)=Yellow (class 0), (0,1)=Green (class 1)
    # (1,0)=Blue (class 2),   (1,1)=White (ignored -> -1)
    img_arr = np.array(
        [
            [[250, 250, 5], [10, 250, 10]],
            [[5, 5, 250], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )

    class_grid = classify_pixels(img_arr, legend=legend, ignore_colors=ignore_colors)

    assert class_grid[0, 0] == 0
    assert class_grid[0, 1] == 1
    assert class_grid[1, 0] == 2
    assert class_grid[1, 1] == -1  # Ignored


def test_digitize_synthetic_image(tmp_path: Path) -> None:
    """Full test of digitize() on a generated test image and YAML metadata."""
    img_file = tmp_path / "test_map.png"
    meta_file = tmp_path / "test_meta.yaml"

    # Create 50x50 green image (class 1: 100-200 mm)
    img = Image.new("RGB", (50, 50), color=(0, 255, 0))
    img.save(img_file)

    meta_content = {
        "target_month": "2020-05",
        "control_points": [
            {"pixel_x": 0.0, "pixel_y": 0.0, "lat": 0.0, "lon": 100.0},
            {"pixel_x": 50.0, "pixel_y": 0.0, "lat": 0.0, "lon": 105.0},
            {"pixel_x": 0.0, "pixel_y": 50.0, "lat": -5.0, "lon": 100.0},
            {"pixel_x": 50.0, "pixel_y": 50.0, "lat": -5.0, "lon": 105.0},
        ],
        "legend": [
            {"label": "Low", "rgb": [255, 255, 0], "lower": 0.0, "upper": 100.0},
            {"label": "Medium", "rgb": [0, 255, 0], "lower": 100.0, "upper": 200.0},
            {"label": "High", "rgb": [0, 0, 255], "lower": 200.0, "upper": 300.0},
        ],
        "ignore_colors": [[255, 255, 255]],
    }
    meta_file.write_text(yaml.dump(meta_content), encoding="utf-8")

    res_df = digitize(img_file, meta_file)
    assert len(res_df) == 1
    row = res_df.iloc[0]
    assert row["source"] == "bmkg_image"
    # Midpoint of 100 and 200 is 150.0
    assert np.isclose(row["precip"], 150.0)
    assert row["quant_error"] == 50.0


def test_extract_table_csv_fallback(tmp_path: Path) -> None:
    csv_file = tmp_path / "sample_season.csv"
    pd.DataFrame(
        {
            "region_id": ["ID-JK"],
            "season_type": ["Kemarau"],
            "onset_dekad": ["Mei II"],
            "characteristic": ["Normal"],
        }
    ).to_csv(csv_file, index=False)

    df = extract_table(csv_file)
    assert len(df) == 1
    assert df["source"].iloc[0] == "bmkg_pdf"
    assert df["season_type"].iloc[0] == "Kemarau"
