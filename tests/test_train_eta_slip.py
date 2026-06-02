from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from etaslip.modeling.dataset import (
    TARGET_MODE_MISSING_OR_SLIP_SECONDS,
    DatasetSpec,
    build_trip_context_from_events,
    load_gold,
    make_dataset_spec,
    merge_trip_context_features,
    merge_vehicle_alert_signals,
    time_split,
)
from etaslip.modeling.trainers import train_logreg, predict_proba


def write_gold_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal gold schema used by load_gold/train
    t = pa.table(
        {
            "feed_ts": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
            "match_status": pa.array(["matched"] * 6, pa.string()),
            "slip_ge_threshold": pa.array([0, 0, 1, 0, 1, 0], pa.int8()),
            "eta_t_minutes": pa.array([10, 12, 15, 8, 20, 9], pa.float32()),
            "top2_headway_sec": pa.array([300, 600, 900, 200, 1200, 250], pa.int32()),
            "num_arrivals_listed": pa.array([3, 2, 2, 4, 2, 3], pa.int16()),
            "dow": pa.array([0, 0, 0, 0, 0, 0], pa.int8()),
            "hour": pa.array([19, 19, 19, 19, 19, 19], pa.int8()),
            "minute": pa.array([0, 1, 2, 3, 4, 5], pa.int8()),
            "stop_id": pa.array(["A", "A", "B", "B", "A", "B"], pa.string()),
        }
    )
    pq.write_table(t, path, compression="zstd")


def test_load_and_train_logreg_smoke(tmp_path: Path):
    spec = DatasetSpec()
    f = tmp_path / "gold.parquet"
    write_gold_parquet(f)

    df = load_gold(f, spec)
    train_df, test_df = time_split(df, train_frac=0.67)

    pipe = train_logreg(train_df, spec)
    prob = predict_proba(pipe, test_df, spec)

    assert len(prob) == len(test_df)
    assert (prob >= 0).all() and (prob <= 1).all()


def test_load_gold_can_synthesize_missing_or_slip_target(tmp_path: Path):
    path = tmp_path / "gold.parquet"
    table = pa.table(
        {
            "feed_ts": pa.array([1, 2, 3], pa.int64()),
            "match_status": pa.array(
                ["matched", "missing_at_tplus", "matched"], pa.string()
            ),
            "slip_ge_threshold": pa.array([0, None, 0], pa.int8()),
            "slip_seconds": pa.array([90, None, 20], pa.float32()),
            "eta_t_minutes": pa.array([10, 12, 15], pa.float32()),
            "top2_headway_sec": pa.array([300, 600, 900], pa.int32()),
            "num_arrivals_listed": pa.array([3, 2, 2], pa.int16()),
            "dow": pa.array([0, 0, 0], pa.int8()),
            "hour": pa.array([19, 19, 19], pa.int8()),
            "minute": pa.array([0, 1, 2], pa.int8()),
            "stop_id": pa.array(["A", "A", "B"], pa.string()),
        }
    )
    pq.write_table(table, path, compression="zstd")
    spec = make_dataset_spec(
        target_mode=TARGET_MODE_MISSING_OR_SLIP_SECONDS,
        slip_threshold_sec=60,
    )

    df = load_gold(path, spec).sort_values("feed_ts").reset_index(drop=True)

    assert df["target"].to_list() == [1, 1, 0]


def test_load_gold_can_add_trip_features(tmp_path: Path):
    path = tmp_path / "gold.parquet"
    table = pa.table(
        {
            "feed_ts": pa.array([1, 61, 121], pa.int64()),
            "match_status": pa.array(["matched", "matched", "matched"], pa.string()),
            "slip_ge_threshold": pa.array([0, 1, 0], pa.int8()),
            "eta_t": pa.array([601, 661, 721], pa.int64()),
            "next_trip_id": pa.array(
                ["113500_6..S01R", "113500_6..S01R", "bad"], pa.string()
            ),
            "eta_t_minutes": pa.array([10, 10, 10], pa.float32()),
            "top2_headway_sec": pa.array([300, 300, 300], pa.int32()),
            "num_arrivals_listed": pa.array([3, 3, 3], pa.int16()),
            "dow": pa.array([0, 0, 0], pa.int8()),
            "hour": pa.array([19, 19, 19], pa.int8()),
            "minute": pa.array([0, 1, 2], pa.int8()),
            "stop_id": pa.array(["640S", "640S", "639S"], pa.string()),
        }
    )
    pq.write_table(table, path, compression="zstd")
    spec = make_dataset_spec(feature_set="history_trip")

    df = load_gold(path, spec).sort_values("feed_ts").reset_index(drop=True)

    assert df["trip_prefix_num"].to_list() == [113500.0, 113500.0, 0.0]
    assert df["trip_pattern"].astype(str).to_list() == ["6..S01R", "6..S01R", ""]


def test_merge_vehicle_alert_signals_adds_model_features():
    df = pa.table(
        {
            "feed_ts": pa.array([1], pa.int64()),
            "next_trip_id": pa.array(["113500_6..S01R"], pa.string()),
            "stop_id": pa.array(["638S"], pa.string()),
            "static_stop_sequence": pa.array([4.0], pa.float64()),
        }
    ).to_pandas()
    signals = pa.table(
        {
            "feed_ts": pa.array([1], pa.int64()),
            "next_trip_id": pa.array(["113500_6..S01R"], pa.string()),
            "vehicle_present": pa.array([1], pa.int8()),
            "vehicle_timestamp": pa.array([None], pa.int64()),
            "vehicle_current_status": pa.array([1], pa.int8()),
            "vehicle_stop_id": pa.array(["637S"], pa.string()),
            "vehicle_stop_sequence": pa.array([3], pa.int32()),
            "trip_alert_delayed": pa.array([1], pa.int8()),
            "vehicle_movement_age_sec": pa.array([45.0], pa.float64()),
        }
    ).to_pandas()

    out = merge_vehicle_alert_signals(df, signals)

    assert out["vehicle_present"].to_list() == [1.0]
    assert out["trip_alert_delayed"].to_list() == [1.0]
    assert out["vehicle_seq_to_selected"].to_list() == [1.0]
    assert out["vehicle_stop_id"].astype(str).to_list() == ["637S"]


def test_trip_context_features_from_events_and_merge():
    selected = pa.table(
        {
            "feed_ts": pa.array([100], pa.int64()),
            "next_trip_id": pa.array(["t1"], pa.string()),
            "stop_id": pa.array(["B"], pa.string()),
            "vehicle_movement_age_sec": pa.array([20.0], pa.float64()),
            "vehicle_stale_90s": pa.array([0.0], pa.float64()),
            "static_progress_pct": pa.array([0.5], pa.float64()),
        }
    ).to_pandas()
    events = pa.table(
        {
            "feed_ts": pa.array([100, 100, 100], pa.int64()),
            "trip_id": pa.array(["t1", "t1", "t1"], pa.string()),
            "stop_id": pa.array(["A", "B", "C"], pa.string()),
            "eta": pa.array([160, 220, 340], pa.int64()),
        }
    ).to_pandas()
    signals = pa.table(
        {
            "feed_ts": pa.array([100], pa.int64()),
            "vehicle_present": pa.array([1], pa.int8()),
            "trip_alert_delayed": pa.array([1], pa.int8()),
            "vehicle_movement_age_sec": pa.array([20.0], pa.float64()),
        }
    ).to_pandas()

    context = build_trip_context_from_events(events, vehicle_alert_signals=signals)
    out = merge_trip_context_features(selected, context)

    assert out["trip_update_stops_remaining"].to_list() == [3.0]
    assert out["trip_stop_rank_remaining"].to_list() == [2.0]
    assert out["trip_eta_gap_prev_stop_sec"].to_list() == [60.0]
    assert out["trip_eta_gap_next_stop_sec"].to_list() == [120.0]
    assert out["route_alert_delayed_trip_count"].to_list() == [1.0]
    assert out["vehicle_age_x_static_progress"].to_list() == [10.0]
