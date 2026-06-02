from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Union
from zoneinfo import ZoneInfo

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

TRIP_NUMERIC_FEATURES: tuple[str, ...] = (
    "trip_prefix_num",
    "trip_prefix_sin",
    "trip_prefix_cos",
)

STATIC_NUMERIC_FEATURES: tuple[str, ...] = (
    "static_offset_sec",
    "static_stop_sequence",
    "static_terminal_offset_sec",
    "static_num_stops",
    "static_remaining_sec",
    "static_progress_pct",
    "static_missing",
    "eta_vs_static_offset_sec",
    "lead_vs_static_remaining_to_stop_sec",
)

VEHICLE_ALERT_NUMERIC_FEATURES: tuple[str, ...] = (
    "vehicle_present",
    "trip_alert_delayed",
    "vehicle_movement_age_sec",
    "vehicle_current_status_num",
    "vehicle_stop_sequence",
    "vehicle_stale_90s",
    "vehicle_seq_to_selected",
    "vehicle_at_selected_stop",
)

TRIP_CONTEXT_NUMERIC_FEATURES: tuple[str, ...] = (
    "trip_update_stops_remaining",
    "trip_update_eta_span_sec",
    "trip_update_first_lead_sec",
    "trip_update_last_lead_sec",
    "trip_stop_rank_remaining",
    "trip_stop_rank_pct_remaining",
    "trip_stops_after_selected",
    "trip_eta_gap_prev_stop_sec",
    "trip_eta_gap_next_stop_sec",
    "route_tripupdate_trip_count",
    "route_stop_update_count",
    "route_eta_mean_lead_sec",
    "route_eta_std_sec",
    "route_vehicle_signal_count",
    "route_alert_delayed_trip_count",
    "route_vehicle_stale_90_count",
    "route_vehicle_movement_age_mean_sec",
    "route_vehicle_movement_age_max_sec",
    "route_vehicle_stale_share",
    "route_alert_delayed_trip_share",
    "vehicle_age_x_trip_rank_pct",
    "vehicle_age_x_static_progress",
    "alert_share_x_vehicle_stale",
    "trip_remaining_per_route_trip",
    "selected_gap_min_stop_sec",
)

CATEGORICAL_FEATURES: tuple[str, ...] = ("stop_id",)
TRIP_CATEGORICAL_FEATURES: tuple[str, ...] = ("trip_pattern",)
VEHICLE_ALERT_CATEGORICAL_FEATURES: tuple[str, ...] = ("vehicle_stop_id",)

STATIC_6_FEATURES_PATH = Path("config/static_6_shape_stop_features.csv")
NY_TZ = ZoneInfo("America/New_York")


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
        elif feature_set == "history_trip":
            numeric_features = (
                BASE_NUMERIC_FEATURES + HISTORY_NUMERIC_FEATURES + TRIP_NUMERIC_FEATURES
            )
            if categorical_features is None:
                categorical_features = CATEGORICAL_FEATURES + TRIP_CATEGORICAL_FEATURES
        elif feature_set == "history_trip_static_vehicle":
            numeric_features = (
                BASE_NUMERIC_FEATURES
                + HISTORY_NUMERIC_FEATURES
                + TRIP_NUMERIC_FEATURES
                + STATIC_NUMERIC_FEATURES
                + VEHICLE_ALERT_NUMERIC_FEATURES
            )
            if categorical_features is None:
                categorical_features = (
                    CATEGORICAL_FEATURES
                    + TRIP_CATEGORICAL_FEATURES
                    + VEHICLE_ALERT_CATEGORICAL_FEATURES
                )
        elif feature_set == "history_trip_static_vehicle_context":
            numeric_features = (
                BASE_NUMERIC_FEATURES
                + HISTORY_NUMERIC_FEATURES
                + TRIP_NUMERIC_FEATURES
                + STATIC_NUMERIC_FEATURES
                + VEHICLE_ALERT_NUMERIC_FEATURES
                + TRIP_CONTEXT_NUMERIC_FEATURES
            )
            if categorical_features is None:
                categorical_features = (
                    CATEGORICAL_FEATURES
                    + TRIP_CATEGORICAL_FEATURES
                    + VEHICLE_ALERT_CATEGORICAL_FEATURES
                )
        else:
            raise ValueError(
                "feature_set must be 'base', 'history', 'history_trip', "
                "'history_trip_static_vehicle', or "
                "'history_trip_static_vehicle_context'"
            )

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


def needs_trip_features(spec: DatasetSpec) -> bool:
    return bool(
        (set(spec.numeric_features) & set(TRIP_NUMERIC_FEATURES))
        or (set(spec.categorical_features) & set(TRIP_CATEGORICAL_FEATURES))
    )


def needs_static_features(spec: DatasetSpec) -> bool:
    return bool(set(spec.numeric_features) & set(STATIC_NUMERIC_FEATURES))


def needs_vehicle_alert_features(spec: DatasetSpec) -> bool:
    return bool(
        (set(spec.numeric_features) & set(VEHICLE_ALERT_NUMERIC_FEATURES))
        or (set(spec.categorical_features) & set(VEHICLE_ALERT_CATEGORICAL_FEATURES))
    )


def needs_trip_context_features(spec: DatasetSpec) -> bool:
    return bool(set(spec.numeric_features) & set(TRIP_CONTEXT_NUMERIC_FEATURES))


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


def add_trip_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse stable, live-available signal from NYCT trip IDs.
    """
    out = df.copy()
    if "next_trip_id" not in out.columns:
        for col in TRIP_NUMERIC_FEATURES:
            out[col] = 0.0
        out["trip_pattern"] = ""
        return out

    trip_id = out["next_trip_id"].fillna("").astype(str)
    prefix = trip_id.map(_trip_prefix_num)
    phase = (prefix % 86400.0) / 86400.0

    out["trip_prefix_num"] = prefix
    out["trip_prefix_sin"] = np.sin(2.0 * np.pi * phase)
    out["trip_prefix_cos"] = np.cos(2.0 * np.pi * phase)
    out["trip_pattern"] = trip_id.map(_trip_pattern).astype("string")
    return out


def _trip_start_sec_from_id(trip_id: pd.Series) -> pd.Series:
    code = trip_id.fillna("").astype(str).map(_trip_prefix_num)
    # NYCT origin-time codes are hundredths of a minute past service-day midnight.
    return code * 0.6


def _local_seconds(ts: pd.Series) -> pd.Series:
    local = pd.to_datetime(ts, unit="s", utc=True, errors="coerce").dt.tz_convert(NY_TZ)
    return ((local.dt.hour * 3600) + (local.dt.minute * 60) + local.dt.second).astype(
        float
    )


def _service_day_delta_seconds(event_sec: pd.Series, start_sec: pd.Series) -> pd.Series:
    delta = event_sec - start_sec
    delta = np.where(delta < -6 * 3600, delta + 86400.0, delta)
    delta = np.where(delta > 30 * 3600, delta - 86400.0, delta)
    return pd.Series(delta, index=event_sec.index)


def _add_trip_timing_context(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "next_trip_id" not in out.columns:
        out["trip_start_sec"] = 0.0
    else:
        out["trip_start_sec"] = _trip_start_sec_from_id(out["next_trip_id"])

    if {"feed_ts", "eta_t"}.issubset(out.columns):
        feed_sec = _local_seconds(pd.to_numeric(out["feed_ts"], errors="coerce"))
        eta_sec = _local_seconds(pd.to_numeric(out["eta_t"], errors="coerce"))
        out["trip_age_sec"] = _service_day_delta_seconds(
            feed_sec, out["trip_start_sec"]
        )
        out["eta_elapsed_since_origin_sec"] = _service_day_delta_seconds(
            eta_sec, out["trip_start_sec"]
        )
        out["lead_sec"] = pd.to_numeric(out["eta_t"], errors="coerce") - pd.to_numeric(
            out["feed_ts"], errors="coerce"
        )
    else:
        out["trip_age_sec"] = 0.0
        out["eta_elapsed_since_origin_sec"] = 0.0
        if "lead_sec" not in out.columns:
            out["lead_sec"] = 0.0

    return out


def load_static_shape_stop_features(
    path: Union[str, Path] = STATIC_6_FEATURES_PATH,
) -> pd.DataFrame:
    p = Path(path)
    columns = [
        "trip_pattern",
        "stop_id",
        "static_offset_sec",
        "static_stop_sequence",
        "static_terminal_offset_sec",
        "static_num_stops",
        "static_remaining_sec",
        "static_progress_pct",
    ]
    if not p.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_csv(p, dtype={"trip_pattern": "string", "stop_id": "string"})


def add_static_features(
    df: pd.DataFrame,
    *,
    static_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    out = _add_trip_timing_context(df)
    if "trip_pattern" not in out.columns:
        out = add_trip_features(out)

    static = (
        load_static_shape_stop_features()
        if static_features is None
        else static_features.copy()
    )
    if not static.empty:
        out = out.merge(static, on=["trip_pattern", "stop_id"], how="left")
    else:
        for col in STATIC_NUMERIC_FEATURES:
            out[col] = np.nan

    out["static_missing"] = out["static_offset_sec"].isna().astype(int)
    for col in (
        "static_offset_sec",
        "static_stop_sequence",
        "static_terminal_offset_sec",
        "static_num_stops",
        "static_remaining_sec",
        "static_progress_pct",
    ):
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(-1.0)

    has_static = out["static_missing"].eq(0)
    out["eta_vs_static_offset_sec"] = np.where(
        has_static,
        out["eta_elapsed_since_origin_sec"] - out["static_offset_sec"],
        0.0,
    )
    out["lead_vs_static_remaining_to_stop_sec"] = np.where(
        has_static,
        out["lead_sec"] - (out["static_offset_sec"] - out["trip_age_sec"]),
        0.0,
    )
    return out


def add_vehicle_alert_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["vehicle_present"] = _numeric_feature(out, "vehicle_present", 0.0)
    out["trip_alert_delayed"] = _numeric_feature(out, "trip_alert_delayed", 0.0)
    out["vehicle_movement_age_sec"] = _numeric_feature(
        out, "vehicle_movement_age_sec", 9999.0
    ).clip(lower=0.0, upper=9999.0)
    out["vehicle_current_status_num"] = _numeric_feature(
        out, "vehicle_current_status", -1.0
    )
    out["vehicle_stop_sequence"] = _numeric_feature(out, "vehicle_stop_sequence", -1.0)
    out["vehicle_stale_90s"] = (
        out["vehicle_present"].eq(1.0) & out["vehicle_movement_age_sec"].gt(90.0)
    ).astype(float)
    if "vehicle_stop_id" in out.columns:
        out["vehicle_stop_id"] = out["vehicle_stop_id"].fillna("").astype("string")
    else:
        out["vehicle_stop_id"] = ""

    if "static_stop_sequence" not in out.columns:
        out["static_stop_sequence"] = -1.0

    out["vehicle_seq_to_selected"] = np.where(
        out["vehicle_stop_sequence"].ge(0.0) & out["static_stop_sequence"].ge(0.0),
        out["static_stop_sequence"] - out["vehicle_stop_sequence"],
        999.0,
    )
    out["vehicle_at_selected_stop"] = (
        out["vehicle_present"].eq(1.0)
        & out["vehicle_stop_id"].astype(str).eq(out["stop_id"].astype(str))
    ).astype(float)
    return out


def _numeric_feature(df: pd.DataFrame, col: str, default: float) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def merge_vehicle_alert_signals(
    df: pd.DataFrame,
    signals: Union[str, Path, pd.DataFrame] | None,
) -> pd.DataFrame:
    out = df.drop(
        columns=[
            "vehicle_present",
            "vehicle_timestamp",
            "vehicle_current_status",
            "vehicle_stop_id",
            "vehicle_stop_sequence",
            "trip_alert_delayed",
            "vehicle_movement_age_sec",
            *VEHICLE_ALERT_NUMERIC_FEATURES,
            *VEHICLE_ALERT_CATEGORICAL_FEATURES,
        ],
        errors="ignore",
    )
    if signals is None:
        return add_vehicle_alert_features(out)

    signal_df = (
        pd.read_parquet(signals) if isinstance(signals, (str, Path)) else signals
    )
    if signal_df.empty:
        return add_vehicle_alert_features(out)

    signal_df = signal_df.drop_duplicates(["feed_ts", "next_trip_id"], keep="last")
    out = out.merge(signal_df, on=["feed_ts", "next_trip_id"], how="left")
    return add_vehicle_alert_features(out)


def build_trip_context_from_events(
    events: pd.DataFrame,
    *,
    vehicle_alert_signals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    columns = ["feed_ts", "next_trip_id", "stop_id", *TRIP_CONTEXT_NUMERIC_FEATURES]
    if events.empty or "eta" not in events.columns:
        return pd.DataFrame(columns=columns)

    out = events.copy()
    if "trip_id" not in out.columns:
        out["trip_id"] = ""
    out["eta"] = pd.to_numeric(out["eta"], errors="coerce")
    out["feed_ts"] = pd.to_numeric(out["feed_ts"], errors="coerce")
    out = out[out["eta"].notna() & out["feed_ts"].notna()].copy()
    out = out[out["eta"] >= out["feed_ts"]].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)

    out["next_trip_id"] = out["trip_id"].fillna("").astype(str)
    out = out.sort_values(["feed_ts", "next_trip_id", "eta", "stop_id"])

    by_trip = out.groupby(["feed_ts", "next_trip_id"], sort=False, observed=True)
    out["trip_update_stops_remaining"] = (
        by_trip["stop_id"].transform("count").astype(float)
    )
    out["trip_update_eta_first"] = by_trip["eta"].transform("min")
    out["trip_update_eta_last"] = by_trip["eta"].transform("max")
    out["trip_update_eta_span_sec"] = (
        out["trip_update_eta_last"] - out["trip_update_eta_first"]
    )
    out["trip_update_first_lead_sec"] = out["trip_update_eta_first"] - out["feed_ts"]
    out["trip_update_last_lead_sec"] = out["trip_update_eta_last"] - out["feed_ts"]
    out["trip_stop_rank_remaining"] = by_trip.cumcount().astype(float) + 1.0
    denom = (out["trip_update_stops_remaining"] - 1.0).clip(lower=1.0)
    out["trip_stop_rank_pct_remaining"] = (
        out["trip_stop_rank_remaining"] - 1.0
    ) / denom
    out["trip_stops_after_selected"] = (
        out["trip_update_stops_remaining"] - out["trip_stop_rank_remaining"]
    )
    out["trip_eta_gap_prev_stop_sec"] = (out["eta"] - by_trip["eta"].shift(1)).fillna(
        9999.0
    )
    out["trip_eta_gap_next_stop_sec"] = (by_trip["eta"].shift(-1) - out["eta"]).fillna(
        9999.0
    )

    by_snapshot = out.groupby("feed_ts", sort=False, observed=True)
    out["route_tripupdate_trip_count"] = by_snapshot["next_trip_id"].transform(
        "nunique"
    )
    out["route_stop_update_count"] = by_snapshot["stop_id"].transform("count")
    out["route_eta_mean_lead_sec"] = (
        by_snapshot["eta"].transform("mean") - out["feed_ts"]
    )
    out["route_eta_std_sec"] = by_snapshot["eta"].transform("std").fillna(0.0)

    snapshot_signals = _route_signal_context(vehicle_alert_signals)
    if not snapshot_signals.empty:
        out = out.merge(snapshot_signals, on="feed_ts", how="left")

    for col in (
        "route_vehicle_signal_count",
        "route_alert_delayed_trip_count",
        "route_vehicle_stale_90_count",
        "route_vehicle_movement_age_mean_sec",
        "route_vehicle_movement_age_max_sec",
    ):
        out[col] = _numeric_feature(out, col, 0.0)
    out["route_vehicle_stale_share"] = out["route_vehicle_stale_90_count"] / out[
        "route_vehicle_signal_count"
    ].clip(lower=1.0)
    out["route_alert_delayed_trip_share"] = out["route_alert_delayed_trip_count"] / out[
        "route_tripupdate_trip_count"
    ].clip(lower=1.0)

    context_cols = [
        "feed_ts",
        "next_trip_id",
        "stop_id",
        *[
            col
            for col in TRIP_CONTEXT_NUMERIC_FEATURES
            if col
            not in {
                "vehicle_age_x_trip_rank_pct",
                "vehicle_age_x_static_progress",
                "alert_share_x_vehicle_stale",
                "trip_remaining_per_route_trip",
                "selected_gap_min_stop_sec",
            }
        ],
    ]
    return out[context_cols].drop_duplicates(
        ["feed_ts", "next_trip_id", "stop_id"], keep="last"
    )


def _route_signal_context(signals: pd.DataFrame | None) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame(columns=["feed_ts"])

    sig = signals.copy()
    sig["vehicle_present"] = _numeric_feature(sig, "vehicle_present", 0.0)
    sig["trip_alert_delayed"] = _numeric_feature(sig, "trip_alert_delayed", 0.0)
    sig["vehicle_movement_age_sec"] = _numeric_feature(
        sig, "vehicle_movement_age_sec", np.nan
    )
    sig["vehicle_stale_90s"] = sig["vehicle_movement_age_sec"].gt(90.0).astype(float)
    return sig.groupby("feed_ts", as_index=False).agg(
        route_vehicle_signal_count=("vehicle_present", "sum"),
        route_alert_delayed_trip_count=("trip_alert_delayed", "sum"),
        route_vehicle_stale_90_count=("vehicle_stale_90s", "sum"),
        route_vehicle_movement_age_mean_sec=("vehicle_movement_age_sec", "mean"),
        route_vehicle_movement_age_max_sec=("vehicle_movement_age_sec", "max"),
    )


def merge_trip_context_features(
    df: pd.DataFrame,
    context: Union[str, Path, pd.DataFrame] | None,
) -> pd.DataFrame:
    out = df.drop(columns=list(TRIP_CONTEXT_NUMERIC_FEATURES), errors="ignore")
    if context is None:
        return add_trip_context_features(out)

    context_df = (
        pd.read_parquet(context) if isinstance(context, (str, Path)) else context.copy()
    )
    if context_df.empty:
        return add_trip_context_features(out)

    context_df = context_df.drop_duplicates(
        ["feed_ts", "next_trip_id", "stop_id"], keep="last"
    )
    out = out.merge(context_df, on=["feed_ts", "next_trip_id", "stop_id"], how="left")
    return add_trip_context_features(out)


def add_trip_context_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    base_cols = [
        col
        for col in TRIP_CONTEXT_NUMERIC_FEATURES
        if col
        not in {
            "vehicle_age_x_trip_rank_pct",
            "vehicle_age_x_static_progress",
            "alert_share_x_vehicle_stale",
            "trip_remaining_per_route_trip",
            "selected_gap_min_stop_sec",
        }
    ]
    for col in base_cols:
        out[col] = _numeric_feature(out, col, -1.0)

    out["vehicle_age_x_trip_rank_pct"] = _numeric_feature(
        out, "vehicle_movement_age_sec", 9999.0
    ) * _numeric_feature(out, "trip_stop_rank_pct_remaining", -1.0)
    out["vehicle_age_x_static_progress"] = _numeric_feature(
        out, "vehicle_movement_age_sec", 9999.0
    ) * _numeric_feature(out, "static_progress_pct", -1.0)
    out["alert_share_x_vehicle_stale"] = _numeric_feature(
        out, "route_alert_delayed_trip_share", -1.0
    ) * _numeric_feature(out, "vehicle_stale_90s", 0.0)
    out["trip_remaining_per_route_trip"] = _numeric_feature(
        out, "trip_update_stops_remaining", -1.0
    ) / _numeric_feature(out, "route_tripupdate_trip_count", 1.0).clip(lower=1.0)
    out["selected_gap_min_stop_sec"] = np.minimum(
        _numeric_feature(out, "trip_eta_gap_prev_stop_sec", 9999.0).clip(upper=9999.0),
        _numeric_feature(out, "trip_eta_gap_next_stop_sec", 9999.0).clip(upper=9999.0),
    )
    for col in TRIP_CONTEXT_NUMERIC_FEATURES:
        out[col] = _numeric_feature(out, col, 0.0)
    return out


def _trip_prefix_num(trip_id: str) -> float:
    match = re.match(r"^(-?\d+)", trip_id)
    return float(match.group(1)) if match else 0.0


def _trip_pattern(trip_id: str) -> str:
    return trip_id.split("_", 1)[1] if "_" in trip_id else ""


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
    out = out.sort_values(["stop_id", "next_trip_id", "feed_ts"]).reset_index(drop=True)

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
    if needs_trip_features(spec):
        out = add_trip_features(out)
    if needs_history_features(spec):
        out = add_history_features(out)
    if needs_static_features(spec):
        out = add_static_features(out)
    if needs_vehicle_alert_features(spec):
        out = add_vehicle_alert_features(out)
    if needs_trip_context_features(spec):
        out = add_trip_context_features(out)

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
    if (
        needs_history_features(spec)
        or needs_trip_features(spec)
        or needs_static_features(spec)
        or needs_trip_context_features(spec)
    ):
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
