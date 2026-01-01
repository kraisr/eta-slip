from __future__ import annotations

import gzip
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.transit import gtfs_realtime_pb2


def to_iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()


def load_snapshot_gz(path: Path) -> bytes:
    with gzip.open(path, "rb") as f:
        return f.read()


def decode_feedmessage(pb: bytes) -> gtfs_realtime_pb2.FeedMessage:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.ParseFromString(pb)
    return msg


@dataclass(frozen=True)
class Arrival:
    eta: int
    trip_id: str


@dataclass(frozen=True)
class SnapshotSummary:
    feed_ts: Optional[int]
    entity_count: int
    tripupdate_count: int
    route_counts: Counter
    # stop_id -> sorted list of arrivals (eta, trip)
    arrivals_by_stop: Dict[str, List[Arrival]]


def summarize_snapshot(path: Path, route_id: str) -> SnapshotSummary:
    pb = load_snapshot_gz(path)
    msg = decode_feedmessage(pb)

    feed_ts = int(msg.header.timestamp) if msg.header.timestamp else None
    route_counts: Counter[str] = Counter()
    tripupdate_count = 0
    arrivals_by_stop: Dict[str, List[Arrival]] = defaultdict(list)

    for ent in msg.entity:
        if not ent.HasField("trip_update"):
            continue
        tripupdate_count += 1
        tu = ent.trip_update

        rid = tu.trip.route_id if (tu.trip and tu.trip.route_id) else "<missing>"
        route_counts[rid] += 1

        if rid != route_id:
            continue

        trip_id = tu.trip.trip_id if (tu.trip and tu.trip.trip_id) else "<missing_trip>"
        for stu in tu.stop_time_update:
            stop_id = stu.stop_id or ""
            if not stop_id:
                continue

            eta = None
            if stu.arrival and stu.arrival.time:
                eta = int(stu.arrival.time)
            elif stu.departure and stu.departure.time:
                eta = int(stu.departure.time)

            if eta is not None:
                arrivals_by_stop[stop_id].append(Arrival(eta=eta, trip_id=trip_id))

    # Sort arrivals per stop
    for stop_id in list(arrivals_by_stop.keys()):
        arrivals_by_stop[stop_id].sort(key=lambda a: a.eta)

    return SnapshotSummary(
        feed_ts=feed_ts,
        entity_count=len(msg.entity),
        tripupdate_count=tripupdate_count,
        route_counts=route_counts,
        arrivals_by_stop=dict(arrivals_by_stop),
    )


def format_summary(summary: SnapshotSummary, route_id: str, top_n: int = 20) -> str:
    lines: List[str] = []
    lines.append(
        f"FeedHeader.timestamp: {summary.feed_ts}"
        + (f" ({to_iso(summary.feed_ts)})" if summary.feed_ts else "")
    )
    lines.append(f"Entities: {summary.entity_count}")
    lines.append(f"TripUpdates: {summary.tripupdate_count}")
    lines.append("")
    lines.append("Top route_ids in snapshot:")
    for rid, c in summary.route_counts.most_common(15):
        lines.append(f"  {rid:>10}  {c}")

    if route_id not in summary.route_counts:
        lines.append("")
        lines.append(f"WARNING: route_id='{route_id}' not seen in this snapshot.")
        return "\n".join(lines)

    # Make a single list: earliest arrival per stop for printing
    rows: List[Tuple[int, str, str, Optional[float]]] = []
    for stop_id, arrivals in summary.arrivals_by_stop.items():
        if not arrivals:
            continue
        first = arrivals[0]
        mins = (
            (first.eta - summary.feed_ts) / 60.0
            if (summary.feed_ts is not None)
            else None
        )
        rows.append((first.eta, stop_id, first.trip_id, mins))

    rows.sort(key=lambda x: x[0])

    lines.append("")
    lines.append(f"Upcoming arrivals for route {route_id} (showing earliest per stop):")
    for eta, stop_id, trip_id, mins in rows[:top_n]:
        mins_s = f"{mins:6.1f} min" if mins is not None else "   ?"
        lines.append(f"  ETA {to_iso(eta)}  ({mins_s})  stop_id={stop_id}  trip_id={trip_id}")

    return "\n".join(lines)
