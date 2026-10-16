"""Data store: Parquet I/O and DuckDB registry for DengueCast.

Manages:
- Reading/writing Parquet files
- Registering tables in DuckDB
- Computing and storing data hashes for run traceability
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import duckdb
import pandas as pd


def compute_df_hash(df: pd.DataFrame) -> str:
    """Compute a deterministic SHA-256 hash for a DataFrame."""
    # Convert dataframe to deterministic bytes
    # Sort columns for consistency
    cols = sorted(df.columns)
    data_bytes = pd.util.hash_pandas_object(df[cols], index=True).values.tobytes()
    return hashlib.sha256(data_bytes).hexdigest()


def write_table(
    df: pd.DataFrame,
    name: str,
    output_dir: str | Path,
    db_path: str | Path | None = None,
) -> str:
    """Write DataFrame to Parquet and register in DuckDB.

    Args:
        df: DataFrame to write.
        name: Table name (e.g., 'cases', 'climate_obs').
        output_dir: Directory for Parquet file.
        db_path: Path to DuckDB file. If None, skip registration.

    Returns:
        Hex hash of the written data for traceability.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parquet_file = out_dir / f"{name}.parquet"
    df.to_parquet(parquet_file, index=False)

    data_hash = compute_df_hash(df)
    hash_file = out_dir / f"{name}.hash"
    hash_file.write_text(data_hash, encoding="utf-8")

    if db_path is not None:
        db_p = Path(db_path)
        db_p.parent.mkdir(parents=True, exist_ok=True)
        # Register in DuckDB
        conn = duckdb.connect(str(db_p))
        try:
            # We create a table or view from the parquet file
            clean_name = "".join(c for c in name if c.isalnum() or c == "_")
            parquet_posix = parquet_file.as_posix()
            conn.execute(
                f"CREATE OR REPLACE TABLE {clean_name} AS SELECT * FROM read_parquet('{parquet_posix}')"
            )
        finally:
            conn.close()

    return data_hash


def read_table(
    name: str,
    data_dir: str | Path,
) -> pd.DataFrame:
    """Read a table from Parquet.

    Args:
        name: Table name.
        data_dir: Directory containing Parquet files.

    Returns:
        DataFrame.
    """
    parquet_file = Path(data_dir) / f"{name}.parquet"
    if not parquet_file.exists():
        raise FileNotFoundError(f"Parquet table '{name}' not found at {parquet_file}")
    return pd.read_parquet(parquet_file)


def get_data_hash(name: str, data_dir: str | Path) -> str:
    """Get stored hash for a table.

    Args:
        name: Table name.
        data_dir: Directory containing Parquet files.

    Returns:
        Hex digest string.
    """
    dir_path = Path(data_dir)
    hash_file = dir_path / f"{name}.hash"
    if hash_file.exists():
        return hash_file.read_text(encoding="utf-8").strip()

    parquet_file = dir_path / f"{name}.parquet"
    if parquet_file.exists():
        df = pd.read_parquet(parquet_file)
        return compute_df_hash(df)

    raise FileNotFoundError(f"Table '{name}' not found in {data_dir}")


def query_duckdb(db_path: str | Path, query: str) -> pd.DataFrame:
    """Execute a read query against DuckDB and return a DataFrame."""
    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        return conn.execute(query).df()
    finally:
        conn.close()
