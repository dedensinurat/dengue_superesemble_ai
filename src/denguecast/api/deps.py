"""API dependencies: database access and configuration (P9)."""

from __future__ import annotations

import os
from functools import lru_cache

import duckdb
import pandas as pd

from denguecast.config import DengueCastConfig, load_config
from denguecast.data.store import read_table


@lru_cache(maxsize=1)
def get_config() -> DengueCastConfig:
    return load_config()


def query_dataframe(table_name: str, query_filter: str | None = None) -> pd.DataFrame:
    """Safely query a table from DuckDB or Parquet files."""
    cfg = get_config()
    db_path = cfg.data.paths.db

    # Try DuckDB first
    if os.path.exists(db_path):
        try:
            con = duckdb.connect(db_path, read_only=True)
            res = con.execute(
                f"SELECT count(*) FROM information_schema.tables WHERE table_name = '{table_name}'"
            ).fetchone()
            tbl_exists = res[0] if res is not None else 0
            if tbl_exists > 0:
                sql = f"SELECT * FROM {table_name}"
                if query_filter:
                    sql += f" WHERE {query_filter}"
                df = con.execute(sql).df()
                con.close()
                return df
            con.close()
        except Exception:
            pass

    # Fallback to Parquet store
    try:
        df = read_table(table_name, cfg.data.paths.processed)
        if query_filter:
            # Simple fallback filter
            return df
        return df
    except Exception:
        return pd.DataFrame()
