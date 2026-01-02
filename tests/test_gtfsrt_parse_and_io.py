import time

from google.transit import gtfs_realtime_pb2

from etaslip.gtfsrt.io import write_snapshot
from etaslip.gtfsrt.parse import parse_feed_timestamp


def make_feedmessage_bytes(ts: int) -> bytes:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.timestamp = ts
    return msg.SerializeToString()


def test_parse_feed_timestamp_reads_header_timestamp():
    ts = 1_700_000_000
    pb = make_feedmessage_bytes(ts)
    assert parse_feed_timestamp(pb) == ts


def test_write_snapshot_uses_dt_and_hour_partitions_no_min(tmp_path):
    ts = 1_700_000_000
    pb = make_feedmessage_bytes(ts)

    out_root = tmp_path / "raw" / "gtfsrt"
    out_path = write_snapshot(out_root, "nyct%2Fgtfs", pb, ts)

    dt = time.strftime("%Y-%m-%d", time.localtime(ts))
    hh = time.strftime("%H", time.localtime(ts))

    assert out_path.parent == out_root / "feed=nyct%2Fgtfs" / f"dt={dt}" / f"hour={hh}"
    assert out_path.name == f"{ts}.pb.gz"
    assert "min=" not in str(out_path)
