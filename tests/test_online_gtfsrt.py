from __future__ import annotations

from google.transit import gtfs_realtime_pb2
import pandas as pd

from etaslip.online.gtfsrt import (
    apply_filters_with_route_fallback,
    parse_tripupdates,
    parse_vehicle_alert_signals,
)


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


def test_parse_vehicle_alert_signals_extracts_vehicle_and_delayed_alert():
    ts = 1_700_000_000
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.header.gtfs_realtime_version = "2.0"
    msg.header.timestamp = ts

    vehicle_ent = msg.entity.add()
    vehicle_ent.id = "v1"
    vehicle = vehicle_ent.vehicle
    vehicle.trip.trip_id = "113500_6..S01R"
    vehicle.trip.route_id = "6"
    vehicle.timestamp = ts - 45
    vehicle.stop_id = "638S"
    vehicle.current_stop_sequence = 4
    vehicle.current_status = gtfs_realtime_pb2.VehiclePosition.STOPPED_AT

    alert_ent = msg.entity.add()
    alert_ent.id = "a1"
    alert = alert_ent.alert
    informed = alert.informed_entity.add()
    informed.trip.trip_id = "113500_6..S01R"
    informed.trip.route_id = "6"
    text = alert.header_text.translation.add()
    text.text = "Train delayed"

    parsed = parse_vehicle_alert_signals(msg.SerializeToString())

    row = parsed.iloc[0]
    assert row["next_trip_id"] == "113500_6..S01R"
    assert row["vehicle_present"] == 1
    assert row["vehicle_movement_age_sec"] == 45
    assert row["trip_alert_delayed"] == 1
