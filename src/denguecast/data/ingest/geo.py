"""Geospatial adapter: GeoJSON loading and adjacency computation."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd


def load_geojson(path: str | Path) -> gpd.GeoDataFrame:
    """Load GeoJSON boundary file.

    Returns:
        GeoDataFrame with region polygons.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"GeoJSON file not found: {p}")
    return gpd.read_file(p)


def compute_adjacency(gdf: gpd.GeoDataFrame, id_col: str = "region_id") -> dict[str, list[str]]:
    """Compute region adjacency via shapely touches.

    Returns:
        Dict mapping region_id -> list of adjacent region_ids (symmetric).
    """
    if id_col not in gdf.columns:
        raise ValueError(
            f"id_col '{id_col}' not present in GeoDataFrame columns: {list(gdf.columns)}"
        )

    adj: dict[str, list[str]] = {str(row[id_col]): [] for _, row in gdf.iterrows()}

    for i, row1 in gdf.iterrows():
        rid1 = str(row1[id_col])
        geom1 = row1.geometry
        for j, row2 in gdf.iterrows():
            if i >= j:
                continue
            rid2 = str(row2[id_col])
            geom2 = row2.geometry
            if geom1 is not None and geom2 is not None and geom1.touches(geom2):
                adj[rid1].append(rid2)
                adj[rid2].append(rid1)

    for rid in adj:
        adj[rid] = sorted(list(set(adj[rid])))

    return adj
