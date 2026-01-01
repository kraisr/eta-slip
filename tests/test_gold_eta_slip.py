from __future__ import annotations

import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from etaslip.pipelines.gold_eta_slip import build_gold_eta_slip_for_files


def write_silver_part(path: Path, feed_ts: int, rows: list[dict]) -> None:
    """
    Create a minimal silver TripUpdates part with required columns:
      feed_ts, route_id, trip_id, stop_id, eta
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = {
        "feed_ts": pa.array([feed_ts] * len(rows), pa.int64()),
        "route_id": pa.array([r["route_id"] for r in rows], pa.string()),
        "trip_id": pa.array([r["trip_id"] for r in rows], pa.string()),
        "stop_id": pa.array([r["stop_id"] for r in rows], pa.string()),
        "eta": pa.array([r["eta"] for r in rows], pa.int64()),
    }
    pq.write_table(pa.table(cols), path, compression="zstd")


def test_gold_eta_slip_matched_label_and_headway(tmp_path: Path):
    # Two snapshots: t0 and t1=t0+300 (exact horizon)
    t0 = 1_700_000_000
    t1 = t0 + 300

    f0 = tmp_path / "silver0.parquet"
    f1 = tmp_path / "silver1.parquet"

    # At t0, stop S has two upcoming trains:
    # tripA arrives in 120s, tripB arrives in 300s -> headway = 180
    write_silver_part(
        f0,
        feed_ts=t0,
        rows=[
            {"route_id": "6", "trip_id": "tripA", "stop_id": "STOP_S", "eta": t0 + 120},
            {"route_id": "6", "trip_id": "tripB", "stop_id": "STOP_S", "eta": t0 + 300},
        ],
    )

    # At t1, same tripA at same stop now predicted at t0+240 (slip=120)
    write_silver_part(
        f1,
        feed_ts=t1,
        rows=[
            {"route_id": "6", "trip_id": "tripA", "stop_id": "STOP_S", "eta": t0 + 240},
        ],
    )

    table, stats = build_gold_eta_slip_for_files(
        [f0, f1],
        horizon_sec=300,
        tolerance_sec=0,
        slip_threshold_sec=120,
        route_filter={"6"},
    )

    assert stats.candidate_examples == 1
    assert stats.matched == 1
    assert table.num_rows == 1

    d = table.to_pydict()
    assert d["stop_id"][0] == "STOP_S"
    assert d["next_trip_id"][0] == "tripA"
    assert d["eta_t"][0] == t0 + 120
    assert d["eta_tplus"][0] == t0 + 240
    assert d["slip_seconds"][0] == 120
    assert d["slip_ge_threshold"][0] == 1
    assert d["top2_headway_sec"][0] == 180
    assert d["match_status"][0] == "matched"


def test_gold_eta_slip_missing_at_tplus(tmp_path: Path):
    t0 = 1_700_000_000
    t1 = t0 + 300

    f0 = tmp_path / "silver0.parquet"
    f1 = tmp_path / "silver1.parquet"

    write_silver_part(
        f0,
        feed_ts=t0,
        rows=[
            {"route_id": "6", "trip_id": "tripA", "stop_id": "STOP_S", "eta": t0 + 120},
        ],
    )
    # At t+Δ snapshot, tripA missing
    write_silver_part(
        f1,
        feed_ts=t1,
        rows=[
            {"route_id": "6", "trip_id": "other", "stop_id": "STOP_S", "eta": t0 + 200},
        ],
    )

    table, stats = build_gold_eta_slip_for_files(
        [f0, f1],
        horizon_sec=300,
        tolerance_sec=0,
        slip_threshold_sec=120,
        route_filter={"6"},
    )

    assert stats.candidate_examples == 1
    assert stats.missing_at_tplus == 1
    assert table.num_rows == 1

    d = table.to_pydict()
    assert d["match_status"][0] == "missing_at_tplus"
    assert d["eta_tplus"][0] is None
    assert d["slip_seconds"][0] is None
    assert d["slip_ge_threshold"][0] is None
