from __future__ import annotations

import numpy as np
import pandas as pd


def baseline_no_slip(df: pd.DataFrame) -> np.ndarray:
    """Always predict probability 0."""
    return np.zeros(len(df), dtype=float)


def baseline_headway_score(df: pd.DataFrame, scale_sec: float = 900.0) -> np.ndarray:
    """
    Continuous score based on headway.
    Example: headway=0 -> 0.0, headway=900 -> 1.0
    """
    x = df["top2_headway_sec"].fillna(0).astype(float).to_numpy()
    return np.clip(x / float(scale_sec), 0.0, 1.0)


def baseline_eta_minutes_score(
    df: pd.DataFrame, min_min: float = 5.0, span_min: float = 20.0
) -> np.ndarray:
    """
    Continuous score based on how far away the ETA is.
    Example: eta<=5min -> 0.0, eta>=25min -> 1.0
    """
    x = df["eta_t_minutes"].fillna(0).astype(float).to_numpy()
    return np.clip((x - float(min_min)) / float(span_min), 0.0, 1.0)
