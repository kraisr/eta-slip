import time
from pathlib import Path

from google.transit import gtfs_realtime_pb2

from etaslip.gtfsrt_ingest import fetch_feed, parse_feed_timestamp, write_snapshot


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


def test_fetch_feed_sets_x_api_key_when_provided(monkeypatch):
    captured = {}

    class DummyResp:
        def __init__(self):
            self.content = b"abc"

        def raise_for_status(self):
            return None

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResp()

    import etaslip.gtfsrt_ingest as ingest_mod
    monkeypatch.setattr(ingest_mod.requests, "get", fake_get)

    out = fetch_feed("https://example.com/feed", api_key="secret123", timeout_sec=20)
    assert out == b"abc"
    assert captured["headers"]["x-api-key"] == "secret123"
    assert captured["timeout"] == 20


def test_fetch_feed_omits_x_api_key_when_none(monkeypatch):
    captured = {}

    class DummyResp:
        def __init__(self):
            self.content = b"xyz"

        def raise_for_status(self):
            return None

    def fake_get(url, headers, timeout):
        captured["headers"] = headers
        return DummyResp()

    import etaslip.gtfsrt_ingest as ingest_mod
    monkeypatch.setattr(ingest_mod.requests, "get", fake_get)

    out = fetch_feed("https://example.com/feed", api_key=None, timeout_sec=20)
    assert out == b"xyz"
    assert "x-api-key" not in captured["headers"]
