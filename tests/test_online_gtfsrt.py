from __future__ import annotations

from google.transit import gtfs_realtime_pb2
import pandas as pd

from etaslip.online.gtfsrt import apply_filters_with_route_fallback, parse_tripupdates


def test_parse_tripupdates_keeps_schedule_relationship_without_eta():
    ts = 1_700_000_000
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.timestamp = ts

    ent = msg.entity.add()
    ent.id = "e1"
    tu = ent.trip_update
    tu.trip.trip_id = "trip_6_a"
    tu.trip.route_id = "6"

    skipped = tu.stop_time_update.add()
    skipped.stop_id = "640S"
    skipped.schedule_relationship = (
        gtfs_realtime_pb2.TripUpdate.StopTimeUpdate.ScheduleRelationship.SKIPPED
    )

    parsed = parse_tripupdates(msg.SerializeToString())

    assert len(parsed) == 1
    assert parsed["stop_id"].iloc[0] == "640S"
    assert parsed["eta"].isna().iloc[0]
    assert parsed["stop_schedule_relationship"].iloc[0] == "SKIPPED"


def test_apply_filters_with_route_fallback_uses_6x_when_6_has_no_southbound_rows():
    events = pd.DataFrame(
        [
            {"route_id": "6", "stop_id": "640N"},
            {"route_id": "6X", "stop_id": "640S"},
        ]
    )

    filtered, routes, used_fallback = apply_filters_with_route_fallback(
        events,
        primary_route_filter={"6"},
        fallback_route_filter={"6X"},
        stop_ids={"640S"},
        stop_suffix="S",
    )

    assert used_fallback is True
    assert routes == {"6X"}
    assert filtered["stop_id"].to_list() == ["640S"]
