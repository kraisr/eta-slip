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
    Returns rows: feed_ts, route_id, trip_id, stop_id, eta (unix seconds),
    plus GTFS-RT schedule relationship fields when available.
    """
    if gtfs_realtime_pb2 is None:
        raise RuntimeError(
            "Missing GTFS-RT protobuf bindings. Add dependency `gtfs-realtime-bindings`."
        )

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
        trip_schedule_relationship = _enum_name(
            gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship,
            getattr(trip, "schedule_relationship", 0),
        )

        for stu in tu.stop_time_update:
            stop_id = getattr(stu, "stop_id", "") or ""
            stop_schedule_relationship = _enum_name(
                gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship,
                getattr(stu, "schedule_relationship", 0),
            )
            eta: int | None = None
            if stu.HasField("arrival") and getattr(stu.arrival, "time", 0):
                eta = int(stu.arrival.time)
            elif stu.HasField("departure") and getattr(stu.departure, "time", 0):
                eta = int(stu.departure.time)

            if not stop_id:
                continue

            rows.append(
                {
                    "feed_ts": feed_ts,
                    "route_id": route_id,
                    "trip_id": trip_id,
                    "stop_id": stop_id,
                    "eta": eta,
                    "trip_schedule_relationship": trip_schedule_relationship,
                    "stop_schedule_relationship": stop_schedule_relationship,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "feed_ts",
                "route_id",
                "trip_id",
                "stop_id",
                "eta",
                "trip_schedule_relationship",
                "stop_schedule_relationship",
            ]
        )

    return pd.DataFrame(rows)


def parse_vehicle_alert_signals(feed_bytes: bytes) -> pd.DataFrame:
    """
    Return per-trip movement and delayed-alert signals from the same subway feed.
    """
    if gtfs_realtime_pb2 is None:
        raise RuntimeError(
            "Missing GTFS-RT protobuf bindings. Add dependency `gtfs-realtime-bindings`."
        )

    msg = gtfs_realtime_pb2.FeedMessage()
    msg.ParseFromString(feed_bytes)

    feed_ts = int(getattr(msg.header, "timestamp", 0) or 0)
    if feed_ts <= 0:
        feed_ts = int(time.time())

    rows: dict[tuple[int, str], dict] = {}
    for ent in msg.entity:
        if ent.HasField("vehicle"):
            vehicle = ent.vehicle
            trip = vehicle.trip
            trip_id = getattr(trip, "trip_id", "") or ""
            if not trip_id:
                continue
            key = (feed_ts, trip_id)
            row = rows.setdefault(key, _empty_signal_row(feed_ts, trip_id))
            row["route_id"] = getattr(trip, "route_id", "") or ""
            row["vehicle_present"] = 1
            row["vehicle_timestamp"] = (
                int(vehicle.timestamp) if getattr(vehicle, "timestamp", 0) else None
            )
            row["vehicle_current_status"] = int(getattr(vehicle, "current_status", -1))
            row["vehicle_stop_id"] = getattr(vehicle, "stop_id", "") or ""
            row["vehicle_stop_sequence"] = (
                int(vehicle.current_stop_sequence)
                if getattr(vehicle, "current_stop_sequence", 0)
                else None
            )

        if ent.HasField("alert"):
            alert = ent.alert
            delayed = _alert_mentions_delay(alert)
            if not delayed:
                continue
            for informed in alert.informed_entity:
                trip = informed.trip
                trip_id = getattr(trip, "trip_id", "") or ""
                if not trip_id:
                    continue
                key = (feed_ts, trip_id)
                row = rows.setdefault(key, _empty_signal_row(feed_ts, trip_id))
                row["route_id"] = (
                    getattr(trip, "route_id", "")
                    or getattr(informed, "route_id", "")
                    or row["route_id"]
                )
                row["trip_alert_delayed"] = 1

    if not rows:
        return _empty_signal_frame()

    out = pd.DataFrame(rows.values())
    out["vehicle_movement_age_sec"] = out["feed_ts"] - pd.to_numeric(
        out["vehicle_timestamp"], errors="coerce"
    )
    return out


def _empty_signal_row(feed_ts: int, trip_id: str) -> dict:
    return {
        "feed_ts": feed_ts,
        "next_trip_id": trip_id,
        "route_id": "",
        "vehicle_present": 0,
        "vehicle_timestamp": None,
        "vehicle_current_status": None,
        "vehicle_stop_id": "",
        "vehicle_stop_sequence": None,
        "trip_alert_delayed": 0,
    }


def _empty_signal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feed_ts",
            "next_trip_id",
            "route_id",
            "vehicle_present",
            "vehicle_timestamp",
            "vehicle_current_status",
            "vehicle_stop_id",
            "vehicle_stop_sequence",
            "trip_alert_delayed",
            "vehicle_movement_age_sec",
        ]
    )


def _alert_mentions_delay(alert) -> bool:
    text = " ".join(t.text for t in alert.header_text.translation).lower()
    return "delay" in text or "delayed" in text


def _enum_name(enum_type, value: int) -> str:
    try:
        return enum_type.Name(int(value))
    except Exception:
        return str(int(value))


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


def apply_filters_with_route_fallback(
    df: pd.DataFrame,
    *,
    primary_route_filter: set[str],
    fallback_route_filter: set[str],
    stop_ids: Optional[set[str]],
    stop_suffix: Optional[str],
) -> tuple[pd.DataFrame, set[str], bool]:
    primary = apply_filters(
        df,
        route_filter=primary_route_filter,
        stop_ids=stop_ids,
        stop_suffix=stop_suffix,
    )
    if not primary.empty:
        return primary, primary_route_filter, False

    fallback = apply_filters(
        df,
        route_filter=fallback_route_filter,
        stop_ids=stop_ids,
        stop_suffix=stop_suffix,
    )
    return fallback, fallback_route_filter, not fallback.empty
