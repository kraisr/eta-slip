import gzip
from pathlib import Path

from google.transit import gtfs_realtime_pb2

from etaslip.gtfsrt.summary import format_summary, summarize_snapshot, to_iso


def write_sample_pb_gz(path: Path, ts: int = 1_700_000_000) -> None:
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
    stu2.arrival.time = ts + 480

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


def test_to_iso_formats_timestamp():
    assert "1970-01-01T00:00:00" in to_iso(0)


def test_summarize_snapshot_includes_route6_and_arrival(tmp_path):
    snap = tmp_path / "sample.pb.gz"
    write_sample_pb_gz(snap, ts=1_700_000_000)

    summary = summarize_snapshot(snap, route_id="6")
    assert summary.feed_ts == 1_700_000_000
    assert "6" in summary.route_counts
    assert "640S" in summary.arrivals_by_stop
    assert summary.arrivals_by_stop["640S"][0].trip_id == "trip_6_a"


def test_format_summary_warns_when_route_missing(tmp_path):
    snap = tmp_path / "sample.pb.gz"
    write_sample_pb_gz(snap, ts=1_700_000_000)

    summary = summarize_snapshot(snap, route_id="Z")
    out = format_summary(summary, route_id="Z", top_n=20)
    assert "WARNING: route_id='Z' not seen in this snapshot." in out
