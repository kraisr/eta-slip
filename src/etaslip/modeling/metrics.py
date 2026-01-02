from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def choose_threshold_max_f1(
    y_true,
    y_prob,
    *,
    min_pred_pos: int = 5,
    default: float = 0.5,
    grid_size: int = 200,
) -> float:
    """
    Choose probability threshold that maximizes F1, with a guardrail that the
    resulting threshold must predict at least `min_pred_pos` positives.

    If no threshold satisfies the guardrail (or there are no positives), returns `default`.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    # If there are no positives, threshold tuning is meaningless.
    if y_true.sum() == 0:
        return float(default)

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    if len(thr) == 0:
        return float(default)

    # f1 has same length as prec/rec; thr is length len(prec)-1
    f1 = (2 * prec * rec) / (prec + rec + 1e-12)

    best_thr = float(default)
    best_f1 = -1.0

    for thr in np.linspace(0.01, 0.99, grid_size):
        pred = (y_prob >= thr).astype(int)
        if int(pred.sum()) < min_pred_pos:
            continue

        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())

        prec = tp / (tp + fp + 1e-12)
        rec = tp / (tp + fn + 1e-12)
        f1 = (2 * prec * rec) / (prec + rec + 1e-12)

        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)

    return best_thr


def metrics_at_threshold(y_true, y_prob, thr: float) -> dict:
    """
    Precision/recall/F1 and confusion counts at a given threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    pred = (y_prob >= float(thr)).astype(int)

    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall + 1e-12)
        if (precision + recall)
        else 0.0
    )

    return {
        "threshold": float(thr),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_classification_metrics(y_true, y_prob) -> dict:
    """
    y_true: 0/1
    y_prob: predicted probability for class 1
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    out: dict[str, Any] = {}

    # PR-AUC is usually the most informative for imbalanced problems
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))

    # ROC-AUC can be computed too (may be optimistic under imbalance)
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        out["roc_auc"] = None

    # A couple operating points
    for thr in (0.5, 0.2, 0.1):
        pred = (y_prob >= thr).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        out[f"precision@{thr}"] = float(prec)
        out[f"recall@{thr}"] = float(rec)

    # Best F1 threshold (for reference; don’t overfocus on it)
    precs, recs, thrs = precision_recall_curve(y_true, y_prob)
    f1 = (2 * precs * recs) / (precs + recs + 1e-12)
    best_i = int(f1.argmax()) if len(f1) else 0
    out["best_f1"] = float(f1[best_i]) if len(f1) else 0.0
    out["best_f1_threshold"] = (
        float(thrs[best_i - 1]) if best_i > 0 and len(thrs) else 0.5
    )

    out["n"] = int(len(y_true))
    out["pos_rate"] = float(y_true.mean()) if len(y_true) else 0.0
    return out
