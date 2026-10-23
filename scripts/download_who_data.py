"""Script to download and save official validated WHO Dengue Global Surveillance data for Indonesia."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd


def download_who_dengue_indonesia(output_path: Path | str = "data/raw/who_dengue_indonesia.csv") -> Path:
    base_url = "https://xmart-api-public.who.int/ARBOV/V_DENGUE_GLOBAL_VALIDATED_PUBLIC"
    query = urllib.parse.urlencode({"$filter": "Country eq 'Indonesia'"})
    url = f"{base_url}?{query}"

    print(f"[*] Mengunduh data resmi WHO dari: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )

    with urllib.request.urlopen(req, timeout=45) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    records = payload.get("value", [])
    if not records:
        raise ValueError("Tidak ada data yang diterima dari API WHO untuk Indonesia.")

    df = pd.DataFrame(records)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Sort chronologically
    if "START_DATE" in df.columns:
        df["START_DATE"] = pd.to_datetime(df["START_DATE"])
        df = df.sort_values("START_DATE").reset_index(drop=True)

    df.to_csv(out_file, index=False)
    print(f"[OK] Berhasil menyimpan {len(df)} rekaman data WHO ke: {out_file.resolve()}")

    # Save download metadata
    meta = {
        "source": "World Health Organization (WHO) Global Dengue Surveillance",
        "api_endpoint": base_url,
        "query": "$filter=Country eq 'Indonesia'",
        "download_timestamp_utc": datetime.now(UTC).isoformat(),
        "total_records": len(df),
        "columns": list(df.columns),
        "date_range": [str(df["START_DATE"].min()), str(df["START_DATE"].max())] if "START_DATE" in df.columns else [],
    }
    meta_path = out_file.with_suffix(".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[OK] Metadata tersimpan di: {meta_path.resolve()}")

    return out_file


if __name__ == "__main__":
    download_who_dengue_indonesia()
