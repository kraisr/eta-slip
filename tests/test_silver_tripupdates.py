import gzip
from pathlib import Path

import pyarrow.parquet as pq
from google.transit import gtfs_realtime_pb2

from etaslip.pipelines.silver_tripupdates import parse_raw_snapshot_to_silver


def write_sample_raw_pb_gz(path: Path, ts: int = 1_700_000_000) -> None:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.timestamp = ts

    e1 = msg.entity.add()
    e1.id = "e1"
    tu1 = e1.trip_update
    tu1.trip.trip_id = "trip_6_a"
    tu1.trip.route_id = "6"

    stu1 = tu1.stop_time_update.add()
    stu1.stop_id = "640S"
    stu1.arrival.time = ts + 300

    stu2 = tu1.stop_time_update.add()
    stu2.stop_id = "640S"
    stu2.departure.time = ts + 360  # departure-only record, should still yield eta

    # Another route to test route filtering
    e2 = msg.entity.add()
    e2.id = "e2"
    tu2 = e2.trip_update
    tu2.trip.trip_id = "trip_other"
    tu2.trip.route_id = "Q"
    stu3 = tu2.stop_time_update.add()
    stu3.stop_id = "Q_STOP"
    stu3.arrival.time = ts + 600

    pb = msg.SerializeToString()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(pb)


def test_parse_raw_snapshot_to_silver_writes_parquet_and_filters_route(tmp_path):
    raw = tmp_path / "raw.pb.gz"
    write_sample_raw_pb_gz(raw, ts=1_700_000_000)

    silver_root = tmp_path / "silver"
    out_path = parse_raw_snapshot_to_silver(
        raw,
        silver_root=silver_root,
        source_feed="nyct%2Fgtfs",
        route_filter={"6"},
        skip_if_exists=False,
    )

    assert out_path is not None
    assert out_path.exists()

    table = pq.read_table(out_path)
    df = table.to_pandas()

    # Only route 6 rows should remain (we created 2 StopTimeUpdates for route 6)
    assert set(df["route_id"].unique()) == {"6"}
    assert len(df) == 2

    # arrival_time present in first, departure_time present in second
    assert df["stop_id"].iloc[0] == "640S"
    assert df["eta"].notna().all()


def test_parse_raw_snapshot_to_silver_returns_none_when_all_filtered(tmp_path):
    raw = tmp_path / "raw.pb.gz"
    write_sample_raw_pb_gz(raw, ts=1_700_000_000)

    silver_root = tmp_path / "silver"
    out_path = parse_raw_snapshot_to_silver(
        raw,
        silver_root=silver_root,
        source_feed="nyct%2Fgtfs",
        route_filter={"Z"},  # filter out everything
        skip_if_exists=False,
    )
    assert out_path is None
