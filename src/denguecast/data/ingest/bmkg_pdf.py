"""BMKG PDF table extractor for seasonal forecast documents (P3d, fallback).

Extracts tabular information from BMKG seasonal outlook publications.
All extracted values represent categorical classifications (e.g. onset dekad,
season characteristic 'Bawah Normal', 'Normal', 'Atas Normal'), NOT continuous numbers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def extract_table(pdf_path: str | Path, config: Any = None) -> pd.DataFrame:
    """Extract tabular data from BMKG PDF seasonal forecast document.

    Args:
        pdf_path: Path to PDF file.
        config: Optional configuration.

    Returns:
        DataFrame with extracted seasonal categories [region_id, season_type, onset_dekad, characteristic, source].
    """
    p = Path(pdf_path)
    if not p.exists():
        raise FileNotFoundError(f"BMKG PDF document not found: {p}")

    # If mock CSV / text provided during tests
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        df["source"] = "bmkg_pdf"
        return df

    try:
        import pdfplumber
    except ImportError:
        logger.warning("pdfplumber is not installed; returning empty DataFrame for PDF.")
        return pd.DataFrame(
            columns=["region_id", "season_type", "onset_dekad", "characteristic", "source"]
        )

    extracted_rows = []
    with pdfplumber.open(p) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for tbl in tables:
                if not tbl or len(tbl) < 2:
                    continue
                header = [str(c).strip() for c in tbl[0]]
                for row in tbl[1:]:
                    if len(row) == len(header):
                        row_dict = dict(zip(header, row, strict=False))
                        row_dict["page"] = page_idx + 1
                        row_dict["source"] = "bmkg_pdf"
                        extracted_rows.append(row_dict)

    if not extracted_rows:
        return pd.DataFrame(
            columns=["region_id", "season_type", "onset_dekad", "characteristic", "source"]
        )

    res = pd.DataFrame(extracted_rows)
    return res
