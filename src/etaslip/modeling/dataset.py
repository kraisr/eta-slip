from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import pandas as pd
import pyarrow.parquet as pq


@dataclass(frozen=True)
class DatasetSpec:
    target_col: str = "slip_ge_threshold"
    match_col: str = "match_status"
    required_match_value: str = "matched"

    # Features available from your gold builder
    numeric_features: tuple[str, ...] = (
        "eta_t_minutes",
        "top2_headway_sec",
        "num_arrivals_listed",
        "dow",
        "hour",
        "minute",
    )
    categorical_features: tuple[str, ...] = ("stop_id",)


def _collect_parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    # treat as glob pattern
    return sorted(Path().glob(str(path)))


def load_gold(path: Union[str, Path], spec: DatasetSpec) -> pd.DataFrame:
    """
    Load one gold Parquet file OR all Parquet files under a directory/glob.

    Filters to match_status == 'matched' and non-null target.
    """
    p = Path(path)
    files = _collect_parquet_files(p)
    if not files:
        raise FileNotFoundError(f"No parquet files found at: {path}")

    dfs: list[pd.DataFrame] = []
    cols = [
        "feed_ts",
        spec.match_col,
        spec.target_col,
        *spec.numeric_features,
        *spec.categorical_features,
        "slip_seconds",
    ]

    for f in files:
        pf = pq.ParquetFile(f)
        available = set(pf.schema_arrow.names)
        use_cols = [c for c in cols if c in available]
        t = pf.read(columns=use_cols)
        dfs.append(t.to_pandas())

    df = pd.concat(dfs, ignore_index=True)

    df = df[df[spec.match_col] == spec.required_match_value]
    df = df[df[spec.target_col].notna()]

    df[spec.target_col] = df[spec.target_col].astype(int)
    for c in spec.numeric_features:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in spec.categorical_features:
        if c in df.columns:
            df[c] = df[c].astype("string")

    needed = list(spec.numeric_features) + list(spec.categorical_features) + [spec.target_col, "feed_ts"]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    return df

def time_split(
    df: pd.DataFrame, *,
    train_frac: float = 0.8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split using feed_ts.
    """
    df = df.sort_values("feed_ts").reset_index(drop=True)
    if df.empty:
        return df, df

    cut = int(len(df) * train_frac)
    cut = max(1, min(cut, len(df) - 1))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()

def _to_utc_datetime(ts: pd.Series) -> pd.Series:
    # Handles int seconds / int ms / datetime-like / strings.
    if np.issubdtype(ts.dtype, np.number):
        # crude but effective: seconds epoch ~1e9, ms epoch ~1e12
        med = float(pd.to_numeric(ts, errors="coerce").dropna().median())
        unit = "ms" if med > 10_000_000_000 else "s"
        return pd.to_datetime(ts, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(ts, utc=True, errors="coerce")


def time_split_3way(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    ts_col: str = "feed_ts",
    split_by: str = "snapshot",  # "snapshot" or "day"
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered split into (train, val, test).

    Key behavior:
    - split_by="snapshot": split on unique feed_ts values (keeps all rows from a snapshot together)
    - split_by="day": split on UTC day derived from feed_ts (best once you have multiple days)
    """
    if train_frac <= 0 or val_frac <= 0 or (train_frac + val_frac) >= 1:
        raise ValueError("train_frac and val_frac must be >0 and sum to <1")
    if ts_col not in df.columns:
        raise ValueError(f"Missing required column: {ts_col}")

    df = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df)
    if n < 3:
        return df.iloc[:0].copy(), df.iloc[:0].copy(), df.copy()

    ts = df[ts_col]
    if split_by == "day":
        dt = _to_utc_datetime(ts)
        key = dt.dt.date
    elif split_by == "snapshot":
        key = ts
    else:
        raise ValueError("split_by must be 'snapshot' or 'day'")

    # Unique keys in time order
    keys = pd.Series(key).dropna().drop_duplicates().to_list()
    m = len(keys)
    if m < 3:
        # not enough distinct snapshots/days to do 3-way split
        return df.iloc[:0].copy(), df.iloc[:0].copy(), df.copy()

    i_train = int(m * train_frac)
    i_val = int(m * (train_frac + val_frac))
    i_train = max(1, min(i_train, m - 2))
    i_val = max(i_train + 1, min(i_val, m - 1))

    train_keys = set(keys[:i_train])
    val_keys = set(keys[i_train:i_val])
    test_keys = set(keys[i_val:])

    train_df = df[pd.Series(key).isin(train_keys)].copy()
    val_df = df[pd.Series(key).isin(val_keys)].copy()
    test_df = df[pd.Series(key).isin(test_keys)].copy()
    return train_df, val_df, test_df