from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def _logit(proba: np.ndarray, *, eps: float) -> np.ndarray:
    p = np.clip(np.asarray(proba, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


@dataclass
class ProbabilityCalibratedModel:
    """
    Thin serving wrapper that preserves an estimator's feature preprocessing while
    calibrating its positive-class probability.
    """

    estimator: Any
    calibrator: LogisticRegression
    method: str = "sigmoid"
    eps: float = 1e-6

    def predict_proba(self, X) -> np.ndarray:
        raw = self.estimator.predict_proba(X)[:, 1]
        z = _logit(raw, eps=self.eps).reshape(-1, 1)
        p = self.calibrator.predict_proba(z)[:, 1]
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def __getattr__(self, name: str):
        estimator = self.__dict__.get("estimator")
        if estimator is None:
            raise AttributeError(name)
        return getattr(estimator, name)


def fit_sigmoid_calibrated_model(
    estimator: Any,
    *,
    y_true,
    y_prob,
    eps: float = 1e-6,
) -> ProbabilityCalibratedModel:
    y = np.asarray(y_true).astype(int)
    if len(np.unique(y)) < 2:
        raise ValueError("Calibration requires both positive and negative examples.")

    z = _logit(np.asarray(y_prob, dtype=float), eps=eps).reshape(-1, 1)
    calibrator = LogisticRegression(solver="lbfgs", max_iter=1000)
    calibrator.fit(z, y)
    return ProbabilityCalibratedModel(
        estimator=estimator,
        calibrator=calibrator,
        method="sigmoid",
        eps=eps,
    )
