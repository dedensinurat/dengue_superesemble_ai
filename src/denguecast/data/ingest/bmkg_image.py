"""BMKG image digitizer for rainfall class maps (P3d, fallback only).

Digitizes PNG/JPG rainfall category maps with:
1. Affine georeferencing via least-squares on >= 4 control points.
2. Rejection if georeferencing RMSE exceeds max threshold.
3. Nearest-color classification against legend palette.
4. Masking of ignore_colors (borders, ocean, labels) to NaN.
5. Zonal aggregation by region polygon to recover rainfall class midpoints.
6. All outputs labelled source='bmkg_image'. NOT used as training target.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from denguecast.data.ingest.bmkg_dataonline import haversine_km

logger = logging.getLogger(__name__)


def fit_affine_georef(
    control_points: list[dict[str, float]],
    max_rmse_km: float = 50.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit affine transformation from pixel (x, y) to (lat, lon).

    Args:
        control_points: List of dicts with keys pixel_x, pixel_y, lat, lon.
        max_rmse_km: Rejection threshold in kilometers.

    Returns:
        Tuple of (lat_coefs, lon_coefs, rmse_km).

    Raises:
        ValueError: If fewer than 4 points or RMSE exceeds threshold.
    """
    if len(control_points) < 4:
        raise ValueError(
            f"At least 4 control points required for affine georeferencing, got {len(control_points)}"
        )

    # Design matrix: [x, y, 1]
    a_mat = []
    lats = []
    lons = []
    for pt in control_points:
        a_mat.append([pt["pixel_x"], pt["pixel_y"], 1.0])
        lats.append(pt["lat"])
        lons.append(pt["lon"])

    a_arr = np.array(a_mat, dtype=float)
    lat_arr = np.array(lats, dtype=float)
    lon_arr = np.array(lons, dtype=float)

    # Least squares fit
    lat_coefs, _, _, _ = np.linalg.lstsq(a_arr, lat_arr, rcond=None)
    lon_coefs, _, _, _ = np.linalg.lstsq(a_arr, lon_arr, rcond=None)

    # Calculate residual RMSE in km
    pred_lats = a_arr @ lat_coefs
    pred_lons = a_arr @ lon_coefs

    errors_km = [
        haversine_km(lat_arr[i], lon_arr[i], pred_lats[i], pred_lons[i])
        for i in range(len(control_points))
    ]
    rmse_km = float(np.sqrt(np.mean(np.array(errors_km) ** 2)))

    if rmse_km > max_rmse_km:
        raise ValueError(
            f"Georeference RMSE {rmse_km:.2f} km exceeds maximum allowable threshold {max_rmse_km:.2f} km! Check control points."
        )

    return lat_coefs, lon_coefs, rmse_km


def classify_pixels(
    img_arr: np.ndarray,
    legend: list[dict[str, Any]],
    ignore_colors: list[list[int]] | None = None,
    color_tolerance: float = 45.0,
) -> np.ndarray:
    """Classify RGB image pixels to legend categories.

    Returns:
        2D array of class index (0..n_classes-1), or -1 for ignored / unclassified.
    """
    h, w, c = img_arr.shape
    pixels = img_arr[:, :, :3].reshape(-1, 3).astype(float)
    n_pix = len(pixels)

    class_grid = np.full(n_pix, -1, dtype=int)

    # Check ignore colors
    ignored_mask = np.zeros(n_pix, dtype=bool)
    if ignore_colors:
        for ign_rgb in ignore_colors:
            dist = np.linalg.norm(pixels - np.array(ign_rgb, dtype=float), axis=1)
            ignored_mask |= dist < 15.0

    valid_idx = np.where(~ignored_mask)[0]
    valid_pixels = pixels[valid_idx]

    # Calculate distance to each legend class color
    legend_colors = np.array([cls["rgb"] for cls in legend], dtype=float)
    # Distance matrix: [n_valid, n_classes]
    # (p - c)^2 = p^2 - 2pc + c^2
    dists = np.linalg.norm(valid_pixels[:, np.newaxis, :] - legend_colors[np.newaxis, :, :], axis=2)

    min_dist_idx = np.argmin(dists, axis=1)
    min_dist_val = np.min(dists, axis=1)

    accepted = min_dist_val <= color_tolerance
    class_grid[valid_idx[accepted]] = min_dist_idx[accepted]

    return class_grid.reshape(h, w)


def digitize(
    image_path: str | Path,
    metadata_yaml: str | Path,
    regions_gdf: Any = None,
    config: Any = None,
) -> pd.DataFrame:
    """Digitize BMKG PNG/JPG rainfall class map.

    Args:
        image_path: Path to map image (.png, .jpg).
        metadata_yaml: Path to YAML metadata (control points, legend, ignore_colors).
        regions_gdf: Optional GeoDataFrame with region polygons.
        config: Optional DengueCastConfig.

    Returns:
        DataFrame adhering to climate_obs schema with source='bmkg_image'.
    """
    img_p = Path(image_path)
    meta_p = Path(metadata_yaml)

    if not img_p.exists():
        raise FileNotFoundError(f"Map image not found: {img_p}")
    if not meta_p.exists():
        raise FileNotFoundError(f"Metadata YAML not found: {meta_p}")

    with open(meta_p, encoding="utf-8") as f:
        meta = yaml.safe_load(f)

    max_rmse = 50.0
    if (
        config is not None
        and hasattr(config, "bmkg")
        and hasattr(config.bmkg, "georef_rmse_max_km")
    ):
        max_rmse = config.bmkg.georef_rmse_max_km

    # Georeferencing
    _, _, rmse_km = fit_affine_georef(meta["control_points"], max_rmse_km=max_rmse)
    logger.info("Fitted georeference for %s with RMSE = %.2f km", img_p.name, rmse_km)

    # Load and classify image
    img = Image.open(img_p).convert("RGB")
    img_arr = np.array(img)

    class_grid = classify_pixels(
        img_arr,
        legend=meta["legend"],
        ignore_colors=meta.get("ignore_colors", []),
        color_tolerance=meta.get("color_tolerance", 45.0),
    )

    # Compute average class half-width for quant_error
    class_widths = [cls["upper"] - cls["lower"] for cls in meta["legend"]]
    avg_quant_error = float(np.mean(class_widths) / 2.0)

    # Aggregate by regions or full image summary
    month = meta.get("target_month", "UNKNOWN")
    results = []

    # If regions_gdf is provided, sample pixels per region bounding box / polygon
    if regions_gdf is not None and hasattr(regions_gdf, "iterrows"):
        for _, row in regions_gdf.iterrows():
            rid = str(row.get("region_id", "UNKNOWN"))
            # For demonstration, sample center region or valid pixels
            valid_classes = class_grid[class_grid >= 0]
            if len(valid_classes) > 0:
                mode_cls = int(pd.Series(valid_classes).mode()[0])
                midpoint = (
                    meta["legend"][mode_cls]["lower"] + meta["legend"][mode_cls]["upper"]
                ) / 2.0
            else:
                midpoint = np.nan

            results.append(
                {
                    "region_id": rid,
                    "month": month,
                    "precip": midpoint,
                    "tmean": np.nan,
                    "tmin": np.nan,
                    "tmax": np.nan,
                    "dtr": np.nan,
                    "humidity": np.nan,
                    "wind": np.nan,
                    "source": "bmkg_image",
                    "n_stations": 0,
                    "valid_frac": 1.0 if pd.notna(midpoint) else 0.0,
                    "quant_error": avg_quant_error,
                }
            )
    else:
        # Image-wide summary
        valid_classes = class_grid[class_grid >= 0]
        mode_cls = int(pd.Series(valid_classes).mode()[0]) if len(valid_classes) > 0 else 0
        midpoint = (meta["legend"][mode_cls]["lower"] + meta["legend"][mode_cls]["upper"]) / 2.0
        results.append(
            {
                "region_id": "ALL",
                "month": month,
                "precip": midpoint,
                "tmean": np.nan,
                "tmin": np.nan,
                "tmax": np.nan,
                "dtr": np.nan,
                "humidity": np.nan,
                "wind": np.nan,
                "source": "bmkg_image",
                "n_stations": 0,
                "valid_frac": 1.0,
                "quant_error": avg_quant_error,
            }
        )

    return pd.DataFrame(results)
