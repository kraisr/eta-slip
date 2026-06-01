from __future__ import annotations

import numpy as np

from etaslip.modeling.metrics import (
    calibration_bins,
    choose_threshold_max_fbeta,
    expected_calibration_error,
    grouped_topk_metrics,
)


def test_calibration_bins_and_ece_are_weighted():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])

    bins = calibration_bins(y, p, n_bins=2)
    ece = expected_calibration_error(y, p, n_bins=2)

    assert len(bins) == 2
    assert bins[0]["actual_rate"] == 0.0
    assert bins[1]["actual_rate"] == 1.0
    assert round(ece, 3) == 0.15


def test_grouped_topk_metrics_selects_per_group():
    groups = np.array(["a", "a", "b", "b"])
    y = np.array([0, 1, 1, 0])
    p = np.array([0.2, 0.8, 0.6, 0.9])

    out = grouped_topk_metrics(groups, y, p, k=1)

    assert out["groups"] == 2
    assert out["selected"] == 2
    assert out["tp"] == 1
    assert out["precision"] == 0.5
    assert out["recall"] == 0.5


def test_choose_threshold_max_fbeta_accepts_precision_weighting():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.8])

    thr = choose_threshold_max_fbeta(y, p, beta=0.5, min_pred_pos=1)

    assert 0.01 <= thr <= 0.99
