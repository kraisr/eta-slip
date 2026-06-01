from __future__ import annotations

import pandas as pd

from etaslip.modeling.dataset import DatasetSpec, make_dataset_spec
from etaslip.online.features import (
    build_features_from_events,
    build_service_issue_summary,
    build_unscored_arrival_fallback,
)


def test_live_features_match_gold_min_lead_semantics():
    feed_ts = 1_700_000_000
    events = pd.DataFrame(
        [
            {"feed_ts": feed_ts, "stop_id": "640S", "eta": feed_ts + 60},
            {"feed_ts": feed_ts, "stop_id": "640S", "eta": feed_ts + 300},
            {"feed_ts": feed_ts, "stop_id": "640S", "eta": feed_ts + 600},
            {"feed_ts": feed_ts, "stop_id": "639S", "eta": feed_ts + 30},
        ]
    )

    feats = build_features_from_events(
        events,
        spec=DatasetSpec(),
        arrival_rank=1,
        min_lead_sec=480,
    )

    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["stop_id"] == "640S"
    assert row["eta_t_minutes"] == 10.0
    assert row["top2_headway_sec"] == 240.0
    assert row["num_arrivals_listed"] == 3


def test_live_features_add_history_when_schema_requires_it():
    spec = make_dataset_spec(feature_set="history")
    history_state: dict = {}
    feed_ts = 1_700_000_000
    first = pd.DataFrame(
        [
            {
                "feed_ts": feed_ts,
                "trip_id": "t1",
                "stop_id": "640S",
                "eta": feed_ts + 600,
            },
            {
                "feed_ts": feed_ts,
                "trip_id": "t2",
                "stop_id": "640S",
                "eta": feed_ts + 900,
            },
        ]
    )
    second = pd.DataFrame(
        [
            {
                "feed_ts": feed_ts + 60,
                "trip_id": "t1",
                "stop_id": "640S",
                "eta": feed_ts + 630,
            },
            {
                "feed_ts": feed_ts + 60,
                "trip_id": "t2",
                "stop_id": "640S",
                "eta": feed_ts + 840,
            },
        ]
    )

    build_features_from_events(
        first,
        spec=spec,
        arrival_rank=1,
        min_lead_sec=480,
        history_state=history_state,
    )
    feats = build_features_from_events(
        second,
        spec=spec,
        arrival_rank=1,
        min_lead_sec=480,
        history_state=history_state,
    )

    row = feats.iloc[0]
    assert row["has_trip_history"] == 1.0
    assert row["history_gap_sec"] == 60.0
    assert row["eta_abs_delta_prev_sec"] == 30.0
    assert row["lead_delta_expected_slip_sec"] == 30.0


def test_unscored_fallback_summarizes_arrivals_and_service_issues():
    feed_ts = 1_700_000_000
    events = pd.DataFrame(
        [
            {
                "feed_ts": feed_ts,
                "stop_id": "640S",
                "eta": feed_ts + 120,
                "trip_schedule_relationship": "SCHEDULED",
                "stop_schedule_relationship": "SCHEDULED",
            },
            {
                "feed_ts": feed_ts,
                "stop_id": "640S",
                "eta": None,
                "trip_schedule_relationship": "SCHEDULED",
                "stop_schedule_relationship": "SKIPPED",
            },
        ]
    )

    fallback = build_unscored_arrival_fallback(
        events,
        stop_order=["640S"],
        stop_name_map={"640S": "Brooklyn Bridge-City Hall"},
        min_lead_sec=480,
    )
    issues = build_service_issue_summary(
        events, stop_name_map={"640S": "Brooklyn Bridge-City Hall"}
    )

    assert fallback["station"].to_list() == ["Brooklyn Bridge-City Hall"]
    assert fallback["next_eta_min"].to_list() == [2.0]
    assert fallback["model_eligible_arrivals"].to_list() == [0]
    assert fallback["service_status"].to_list() == ["SKIPPED"]
    assert issues["service_status"].to_list() == ["SKIPPED"]
