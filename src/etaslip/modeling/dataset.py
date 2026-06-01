from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TARGET_MODE_COLUMN = "column"
TARGET_MODE_SLIP_SECONDS = "slip_ge_seconds"
TARGET_MODE_MISSING_OR_SLIP_SECONDS = "missing_or_slip_ge_seconds"

BASE_NUMERIC_FEATURES: tuple[str, ...] = (
    "eta_t_minutes",
    "top2_headway_sec",
    "num_arrivals_listed",
    "dow",
    "hour",
    "minute",
    "minute_of_day",
    "is_weekend",
    "headway_eta_ratio",
    "log_headway_sec",
)

HISTORY_NUMERIC_FEATURES: tuple[str, ...] = (
    "has_trip_history",
    "fresh_trip_history_10m",
    "history_gap_sec",
    "eta_abs_delta_prev_sec",
    "lead_delta_prev_sec",
    "lead_delta_expected_slip_sec",
    "eta_abs_delta_mean3_sec",
    "eta_abs_delta_max3_sec",
    "lead_delta_mean3_sec",
    "lead_delta_max3_sec",
    "station_eta_abs_delta_mean_sec",
    "station_eta_abs_delta_max_sec",
    "station_lead_delta_mean_sec",
    "station_lead_delta_max_sec",
    "station_share_positive_eta_delta",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("stop_id",)


@dataclass(frozen=True)
class DatasetSpec:
    target_col: str = "slip_ge_threshold"
    match_col: str = "match_status"
    required_match_value: str = "matched"
    target_mode: str = TARGET_MODE_COLUMN
    slip_threshold_sec: int = 120

    # Features available from your gold builder
    numeric_features: tuple[str, ...] = BASE_NUMERIC_FEATURES
    categorical_features: tuple[str, ...] = CATEGORICAL_FEATURES


def make_dataset_spec(
    *,
    target_mode: str = TARGET_MODE_COLUMN,
    slip_threshold_sec: int = 120,
    feature_set: str = "base",
    numeric_features: tuple[str, ...] | None = None,
    categorical_features: tuple[str, ...] | None = None,
) -> DatasetSpec:
    if numeric_features is None:
        if feature_set == "base":
            numeric_features = BASE_NUMERIC_FEATURES
        elif feature_set == "history":
            numeric_features = BASE_NUMERIC_FEATURES + HISTORY_NUMERIC_FEATURES
        else:
            raise ValueError("feature_set must be 'base' or 'history'")

    target_col = "slip_ge_threshold" if target_mode == TARGET_MODE_COLUMN else "target"
    return DatasetSpec(
        target_col=target_col,
        target_mode=target_mode,
        slip_threshold_sec=int(slip_threshold_sec),
        numeric_features=tuple(numeric_features),
        categorical_features=tuple(categorical_features or CATEGORICAL_FEATURES),
    )


def needs_history_features(spec: DatasetSpec) -> bool:
    return bool(set(spec.numeric_features) & set(HISTORY_NUMERIC_FEATURES))


def _collect_parquet_files(path: Path) -> list[Path]:
    if path.is_file() and path.suffix == ".parquet":
        return [path]
    if path.is_dir():
        return sorted(path.rglob("*.parquet"))
    # treat as glob pattern
    return sorted(Path().glob(str(path)))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add deterministic features that can be built both offline and live.
    """
    out = df.copy()
    if {"hour", "minute"}.issubset(out.columns):
        hour = pd.to_numeric(out["hour"], errors="coerce")
        minute = pd.to_numeric(out["minute"], errors="coerce")
        out["minute_of_day"] = hour * 60.0 + minute

    if "dow" in out.columns:
        dow = pd.to_numeric(out["dow"], errors="coerce")
        out["is_weekend"] = dow.isin([5, 6]).astype(int)

    if {"top2_headway_sec", "eta_t_minutes"}.issubset(out.columns):
        headway = pd.to_numeric(out["top2_headway_sec"], errors="coerce")
        eta_sec = pd.to_numeric(out["eta_t_minutes"], errors="coerce") * 60.0
        out["headway_eta_ratio"] = headway / eta_sec.clip(lower=1.0)
        out["log_headway_sec"] = np.log1p(headway.clip(lower=0.0))

    return out


def add_history_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add previous-snapshot features for the same stop/trip and current station.

    These are leakage-safe for model training because each row only uses values
    available at or before the current feed timestamp.
    """
    out = df.copy()
    if not {"feed_ts", "eta_t", "stop_id", "next_trip_id"}.issubset(out.columns):
        for col in HISTORY_NUMERIC_FEATURES:
            out[col] = _history_default(col)
        return out

    out["feed_ts"] = pd.to_numeric(out["feed_ts"], errors="coerce")
    out["eta_t"] = pd.to_numeric(out["eta_t"], errors="coerce")
    out["lead_sec"] = out["eta_t"] - out["feed_ts"]
    out = out.sort_values(["stop_id", "next_trip_id", "feed_ts"]).reset_index(
        drop=True
    )

    by_trip_stop = out.groupby(["stop_id", "next_trip_id"], sort=False, observed=True)
    out["prev_feed_ts"] = by_trip_stop["feed_ts"].shift(1)
    out["prev_eta_t"] = by_trip_stop["eta_t"].shift(1)
    out["prev_lead_sec"] = by_trip_stop["lead_sec"].shift(1)

    out["history_gap_sec"] = out["feed_ts"] - out["prev_feed_ts"]
    out["eta_abs_delta_prev_sec"] = out["eta_t"] - out["prev_eta_t"]
    out["lead_delta_prev_sec"] = out["lead_sec"] - out["prev_lead_sec"]
    out["has_trip_history"] = out["prev_feed_ts"].notna().astype(int)
    out["fresh_trip_history_10m"] = out["history_gap_sec"].between(1, 600).astype(int)
    out["lead_delta_expected_slip_sec"] = (
        out["lead_delta_prev_sec"] + out["history_gap_sec"]
    )

    for col in ("eta_abs_delta_prev_sec", "lead_delta_prev_sec"):
        out[f"{col}_mean3"] = by_trip_stop[col].transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
        out[f"{col}_max3"] = by_trip_stop[col].transform(
            lambda s: s.rolling(3, min_periods=1).max()
        )

    out = out.rename(
        columns={
            "eta_abs_delta_prev_sec_mean3": "eta_abs_delta_mean3_sec",
            "eta_abs_delta_prev_sec_max3": "eta_abs_delta_max3_sec",
            "lead_delta_prev_sec_mean3": "lead_delta_mean3_sec",
            "lead_delta_prev_sec_max3": "lead_delta_max3_sec",
        }
    )

    by_stop_snapshot = out.groupby(["stop_id", "feed_ts"], sort=False, observed=True)
    out["station_eta_abs_delta_mean_sec"] = by_stop_snapshot[
        "eta_abs_delta_prev_sec"
    ].transform("mean")
    out["station_eta_abs_delta_max_sec"] = by_stop_snapshot[
        "eta_abs_delta_prev_sec"
    ].transform("max")
    out["station_lead_delta_mean_sec"] = by_stop_snapshot[
        "lead_delta_prev_sec"
    ].transform("mean")
    out["station_lead_delta_max_sec"] = by_stop_snapshot[
        "lead_delta_prev_sec"
    ].transform("max")
    out["station_share_positive_eta_delta"] = by_stop_snapshot[
        "eta_abs_delta_prev_sec"
    ].transform(lambda s: (s > 0).mean())

    for col in HISTORY_NUMERIC_FEATURES:
        out[col] = (
            pd.to_numeric(out[col], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(_history_default(col))
        )
    out["history_gap_sec"] = out["history_gap_sec"].clip(upper=9999.0)
    return out


def _history_default(col: str) -> float:
    return 9999.0 if col == "history_gap_sec" else 0.0


def apply_target(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    out = df.copy()

    if spec.target_mode == TARGET_MODE_COLUMN:
        out = out[out[spec.match_col] == spec.required_match_value]
        out = out[out[spec.target_col].notna()]
        out[spec.target_col] = out[spec.target_col].astype(int)
        return out

    if "slip_seconds" not in out.columns:
        raise ValueError("Synthetic target modes require a slip_seconds column")

    out["slip_seconds"] = pd.to_numeric(out["slip_seconds"], errors="coerce")
    if spec.target_mode == TARGET_MODE_SLIP_SECONDS:
        out = out[out[spec.match_col] == spec.required_match_value].copy()
        out = out[out["slip_seconds"].notna()].copy()
        out[spec.target_col] = (
            out["slip_seconds"] >= float(spec.slip_threshold_sec)
        ).astype(int)
        return out

    if spec.target_mode == TARGET_MODE_MISSING_OR_SLIP_SECONDS:
        out = out[
            out[spec.match_col].isin([spec.required_match_value, "missing_at_tplus"])
        ].copy()
        out[spec.target_col] = (
            (out[spec.match_col] == "missing_at_tplus")
            | (out["slip_seconds"] >= float(spec.slip_threshold_sec))
        ).astype(int)
        return out

    raise ValueError(f"Unknown target_mode: {spec.target_mode}")


def prepare_gold_dataframe(df: pd.DataFrame, spec: DatasetSpec) -> pd.DataFrame:
    out = add_derived_features(df)
    if needs_history_features(spec):
        out = add_history_features(out)

    out = apply_target(out, spec)

    for c in spec.numeric_features:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    for c in spec.categorical_features:
        if c in out.columns:
            out[c] = out[c].astype("string")

    needed = (
        list(spec.numeric_features)
        + list(spec.categorical_features)
        + [spec.target_col, "feed_ts"]
    )
    out = out.dropna(subset=[c for c in needed if c in out.columns])
    out[spec.target_col] = out[spec.target_col].astype(int)
    return out


def load_gold(path: Union[str, Path], spec: DatasetSpec) -> pd.DataFrame:
    """
    Load one gold Parquet file OR all Parquet files under a directory/glob.

    Applies the DatasetSpec target definition and drops rows missing required
    features.
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
    if needs_history_features(spec):
        cols.extend(["next_trip_id", "eta_t"])

    for f in files:
        pf = pq.ParquetFile(f)
        available = set(pf.schema_arrow.names)
        use_cols = [c for c in cols if c in available]
        t = pf.read(columns=use_cols)
        dfs.append(t.to_pandas())

    df = pd.concat(dfs, ignore_index=True)
    return prepare_gold_dataframe(df, spec)


def time_split(
    df: pd.DataFrame,
    *,
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
