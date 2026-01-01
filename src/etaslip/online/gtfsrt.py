from __future__ import annotations

import gzip
import time
from typing import Optional

import pandas as pd
import requests

# Protobuf bindings
try:
    from google.transit import gtfs_realtime_pb2  # type: ignore
except Exception:  # pragma: no cover
    gtfs_realtime_pb2 = None  # type: ignore


def maybe_gunzip(blob: bytes) -> bytes:
    if len(blob) >= 2 and blob[0] == 0x1F and blob[1] == 0x8B:
        return gzip.decompress(blob)
    return blob


def fetch_feed_bytes(feed_url: str, timeout_sec: int = 20) -> bytes:
    r = requests.get(feed_url, timeout=timeout_sec)
    r.raise_for_status()
    return r.content


def parse_tripupdates(feed_bytes: bytes) -> pd.DataFrame:
    """
    Returns rows: feed_ts, route_id, trip_id, stop_id, eta (unix seconds)
    """
    if gtfs_realtime_pb2 is None:
        raise RuntimeError("Missing GTFS-RT protobuf bindings. Add dependency `gtfs-realtime-bindings`.")

    msg = gtfs_realtime_pb2.FeedMessage()
    msg.ParseFromString(feed_bytes)

    feed_ts = int(getattr(msg.header, "timestamp", 0) or 0)
    if feed_ts <= 0:
        feed_ts = int(time.time())

    rows: list[dict] = []
    for ent in msg.entity:
        if not ent.HasField("trip_update"):
            continue
        tu = ent.trip_update
        trip = tu.trip

        route_id = getattr(trip, "route_id", "") or ""
        trip_id = getattr(trip, "trip_id", "") or ""

        for stu in tu.stop_time_update:
            stop_id = getattr(stu, "stop_id", "") or ""
            eta = 0
            if stu.HasField("arrival") and getattr(stu.arrival, "time", 0):
                eta = int(stu.arrival.time)
            elif stu.HasField("departure") and getattr(stu.departure, "time", 0):
                eta = int(stu.departure.time)

            if not stop_id or eta <= 0:
                continue

            rows.append(
                {
                    "feed_ts": feed_ts,
                    "route_id": route_id,
                    "trip_id": trip_id,
                    "stop_id": stop_id,
                    "eta": eta,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["feed_ts", "route_id", "trip_id", "stop_id", "eta"])

    return pd.DataFrame(rows)


def apply_filters(
    df: pd.DataFrame,
    *,
    route_filter: Optional[set[str]],
    stop_ids: Optional[set[str]],
    stop_suffix: Optional[str],
) -> pd.DataFrame:
    out = df.copy()
    if route_filter:
        out = out[out["route_id"].isin(route_filter)]
    if stop_ids:
        out = out[out["stop_id"].isin(stop_ids)]
    if stop_suffix:
        out = out[out["stop_id"].astype(str).str.endswith(stop_suffix)]
    return out
