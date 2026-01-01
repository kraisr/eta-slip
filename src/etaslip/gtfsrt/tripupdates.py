from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterable, Optional

from google.transit import gtfs_realtime_pb2


@dataclass(frozen=True)
class StopTimeRow:
    # metadata
    source_feed: str
    ingest_ts: int  # when we processed the file (epoch seconds)

    # from FeedHeader
    feed_ts: int

    # TripDescriptor / TripUpdate
    trip_id: str
    route_id: str

    # StopTimeUpdate
    stop_id: str
    stop_sequence: Optional[int]
    arrival_time: Optional[int]
    departure_time: Optional[int]
    eta: Optional[int]  # arrival_time preferred, else departure_time
    delay_seconds: Optional[int]
    stu_schedule_relationship: Optional[str]
    trip_schedule_relationship: Optional[str]


def _schedule_relationship_name(val: int) -> str:
    # Works for both TripDescriptor and StopTimeUpdate schedule_relationship enums
    # (they share names like SCHEDULED / SKIPPED / NO_DATA, etc)
    return gtfs_realtime_pb2.TripDescriptor.ScheduleRelationship.Name(val)


def iter_stop_time_rows(
    msg: gtfs_realtime_pb2.FeedMessage,
    *,
    source_feed: str,
    route_filter: Optional[set[str]] = None,
    require_stop_id: bool = True,
    ingest_ts: Optional[int] = None,
) -> Iterable[StopTimeRow]:
    """
    Flatten a FeedMessage into stop-level rows.

    - One row per StopTimeUpdate
    - eta = arrival.time if present else departure.time
    """
    if not msg.header.timestamp:
        # If header timestamp is missing, treat as unprocessable for silver.
        return

    feed_ts = int(msg.header.timestamp)
    ingest_ts = int(ingest_ts or time.time())

    for ent in msg.entity:
        if not ent.HasField("trip_update"):
            continue

        tu = ent.trip_update
        trip = tu.trip

        route_id = trip.route_id or ""
        if route_filter is not None and route_id not in route_filter:
            continue

        trip_id = trip.trip_id or ""

        # trip schedule_relationship is optional
        trip_sr = None
        if trip and trip.schedule_relationship is not None:
            try:
                trip_sr = _schedule_relationship_name(trip.schedule_relationship)
            except Exception:
                trip_sr = str(int(trip.schedule_relationship))

        for stu in tu.stop_time_update:
            stop_id = stu.stop_id or ""
            if require_stop_id and not stop_id:
                continue

            stop_sequence = int(stu.stop_sequence) if stu.stop_sequence else None

            arrival_time = int(stu.arrival.time) if (stu.arrival and stu.arrival.time) else None
            departure_time = int(stu.departure.time) if (stu.departure and stu.departure.time) else None
            eta = arrival_time if arrival_time is not None else departure_time

            delay_seconds = None
            # delay is optional and can appear on arrival/departure
            if stu.arrival and stu.arrival.delay:
                delay_seconds = int(stu.arrival.delay)
            elif stu.departure and stu.departure.delay:
                delay_seconds = int(stu.departure.delay)

            stu_sr = None
            if stu and stu.schedule_relationship is not None:
                try:
                    # StopTimeUpdate.ScheduleRelationship has same naming in proto
                    stu_sr = gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.Name(
                        stu.schedule_relationship
                    )
                except Exception:
                    stu_sr = str(int(stu.schedule_relationship))

            yield StopTimeRow(
                source_feed=source_feed,
                ingest_ts=ingest_ts,
                feed_ts=feed_ts,
                trip_id=trip_id,
                route_id=route_id,
                stop_id=stop_id,
                stop_sequence=stop_sequence,
                arrival_time=arrival_time,
                departure_time=departure_time,
                eta=eta,
                delay_seconds=delay_seconds,
                stu_schedule_relationship=stu_sr,
                trip_schedule_relationship=trip_sr,
            )
