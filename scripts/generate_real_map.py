"""CLI runner script for generating real dengue distribution map in Indonesia.

Outputs:
1. reports/visualizations/peta_penyebaran_dbd_indonesia.html (Interactive HTML Dashboard)
2. reports/visualizations/peta_penyebaran_dbd_indonesia.png (High-Resolution Static Map 300 DPI)
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

# Add project root and src to path if needed
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from denguecast.visualizations.real_map import (  # noqa: E402
    DengueMapConfig,
    generate_high_res_static_map,
    generate_interactive_html_map,
    load_real_dengue_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Buat visualisasi peta real penyebaran kasus DBD di 34 Provinsi Indonesia beserta kategori warnanya."
    )
    parser.add_argument(
        "--output-html",
        type=Path,
        default=Path("reports/visualizations/peta_penyebaran_dbd_indonesia.html"),
        help="Path tujuan file HTML interaktif",
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("reports/visualizations/peta_penyebaran_dbd_indonesia.png"),
        help="Path tujuan file PNG resolusi tinggi",
    )
    args = parser.parse_args()

    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        with contextlib.suppress(Exception):
            sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    print("=" * 70)
    print("[*] SISTEM VISUALISASI PETA REAL PENYEBARAN DBD DI INDONESIA - DENGUECAST")
    print("=" * 70)

    cfg = DengueMapConfig(
        output_html=args.output_html,
        output_png=args.output_png,
    )

    print("\n[1/3] Memuat dan memproses data riil epidemiologi (OpenDengue v1.3 & BPS)...")
    summary_df, geo_df, meta = load_real_dengue_data(cfg)
    print(f"      -> Berhasil memproses {len(summary_df)} provinsi Indonesia.")
    print(f"      -> Total Kasus Riil Nasional: {meta['total_national_cases']:,} kasus ({meta['period']}).")
    print(f"      -> Top 1 Daerah: {summary_df.iloc[0]['name']} ({summary_df.iloc[0]['total_cases']:,} kasus).")

    print("\n[2/3] Membuat Peta Interaktif HTML (Choropleth + Centroid Marker)...")
    html_path = generate_interactive_html_map(
        summary_df,
        geo_df,
        output_path=cfg.output_html,
        geojson_path=cfg.geojson_path,
        world_geojson_path=cfg.world_geojson_path,
    )
    print(f"      -> File HTML tersimpan di: {html_path.resolve()}")

    # Sync to public/index.html for deployment (GitHub Pages, Vercel, Netlify)
    public_index = ROOT_DIR / "public" / "index.html"
    public_index.parent.mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy2(html_path, public_index)
    print(f"      -> Berhasil disinkronisasi ke: {public_index.resolve()} (Deploy Target)")

    print("\n[3/3] Membuat Peta Publikasi Resolusi Tinggi (PNG 300 DPI)...")
    png_path = generate_high_res_static_map(
        summary_df,
        geo_df,
        output_path=cfg.output_png,
    )
    print(f"      -> File PNG tersimpan di: {png_path.resolve()}")

    print("\n" + "=" * 70)
    print("[OK] Selesai! Visualisasi peta real penyebaran DBD Indonesia berhasil dibuat.")
    print("=" * 70)


if __name__ == "__main__":
    main()
