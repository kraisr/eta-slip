from __future__ import annotations

import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

from etaslip.modeling.dataset import DatasetSpec

_PERF_PANEL = Path(__file__).resolve().parents[1] / "app" / "perf_panel.py"
_SPEC = importlib.util.spec_from_file_location("perf_panel", _PERF_PANEL)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_eval_precision_by_day = _MODULE._eval_precision_by_day


class AlwaysAlertModel:
    def predict_proba(self, X):
        return np.column_stack(
            [np.zeros(len(X), dtype=float), np.ones(len(X), dtype=float)]
        )


def test_eval_precision_by_day_derives_missing_feature_columns():
    feed_ts = 1_700_000_000
    df = pd.DataFrame(
        [
            {
                "feed_ts": feed_ts,
                "match_status": "matched",
                "slip_ge_threshold": 1,
                "eta_t_minutes": 10.0,
                "top2_headway_sec": 240.0,
                "num_arrivals_listed": 3,
                "dow": 0,
                "hour": 8,
                "minute": 30,
                "stop_id": "640S",
            },
            {
                "feed_ts": feed_ts + 60,
                "match_status": "matched",
                "slip_ge_threshold": 0,
                "eta_t_minutes": 12.0,
                "top2_headway_sec": 300.0,
                "num_arrivals_listed": 4,
                "dow": 0,
                "hour": 8,
                "minute": 31,
                "stop_id": "639S",
            },
        ]
    )

    out = _eval_precision_by_day(
        df,
        pipe=AlwaysAlertModel(),
        spec=DatasetSpec(),
        thr=0.5,
        model_name="xgb",
    )

    assert not out.empty
    assert out["alerts"].iloc[0] == 2
    assert out["precision"].iloc[0] == 0.5
