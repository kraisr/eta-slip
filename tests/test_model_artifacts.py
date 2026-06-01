from __future__ import annotations

import json
from pathlib import Path

from etaslip.online.model_artifacts import load_dataset_spec, load_serving_params


def test_load_serving_params_defaults_for_older_artifacts(tmp_path: Path):
    (tmp_path / "feature_schema.json").write_text(json.dumps({"target": "x"}))

    params = load_serving_params(tmp_path)

    assert params.arrival_rank == 1
    assert params.min_lead_sec == 480


def test_load_serving_params_reads_schema(tmp_path: Path):
    (tmp_path / "feature_schema.json").write_text(
        json.dumps({"serving": {"arrival_rank": 3, "min_lead_sec": 600}})
    )

    params = load_serving_params(tmp_path)

    assert params.arrival_rank == 3
    assert params.min_lead_sec == 600


def test_load_dataset_spec_reads_schema_target_and_features(tmp_path: Path):
    (tmp_path / "feature_schema.json").write_text(
        json.dumps(
            {
                "numeric_features": ["eta_t_minutes", "has_trip_history"],
                "categorical_features": ["stop_id"],
                "feature_set": "history",
                "labeling": {
                    "target_mode": "missing_or_slip_ge_seconds",
                    "slip_threshold_sec": 60,
                },
            }
        )
    )

    spec = load_dataset_spec(tmp_path)

    assert spec.target_col == "target"
    assert spec.target_mode == "missing_or_slip_ge_seconds"
    assert spec.slip_threshold_sec == 60
    assert spec.numeric_features == ("eta_t_minutes", "has_trip_history")
