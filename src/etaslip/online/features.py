from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from etaslip.modeling.dataset import DatasetSpec
from etaslip.online.constants import NY_TZ


def build_features_from_events(
    events: pd.DataFrame,
    *,
    spec: DatasetSpec,
    arrival_rank: int,
    min_lead_sec: int,
    tz: ZoneInfo = NY_TZ,
) -> pd.DataFrame:
    """
    One row per stop_id:
      eta_t_minutes, top2_headway_sec, num_arrivals_listed, dow, hour, minute, stop_id
    """
    if events.empty:
        return pd.DataFrame(
            columns=list(spec.numeric_features) + list(spec.categorical_features)
        )

    feed_ts = int(events["feed_ts"].iloc[0])
    feed_dt = datetime.fromtimestamp(feed_ts, tz=tz)

    e = events.copy()
    e["lead_sec"] = e["eta"].astype(int) - int(feed_ts)

    e = e[e["lead_sec"] > 0]
    if min_lead_sec > 0:
        e = e[e["lead_sec"] >= int(min_lead_sec)]

    if e.empty:
        return pd.DataFrame(
            columns=list(spec.numeric_features) + list(spec.categorical_features)
        )

    rows: list[dict] = []
    for stop_id, g in e.groupby("stop_id", sort=False):
        leads = np.sort(g["lead_sec"].to_numpy(dtype=float))
        num = int(len(leads))
        if num < arrival_rank:
            continue

        lead_k = float(leads[arrival_rank - 1])
        headway = float(leads[1] - leads[0]) if num >= 2 else 0.0

        rows.append(
            {
                "stop_id": str(stop_id),
                "eta_t_minutes": lead_k / 60.0,
                "top2_headway_sec": headway,
                "num_arrivals_listed": num,
                "dow": int(feed_dt.weekday()),
                "hour": int(feed_dt.hour),
                "minute": int(feed_dt.minute),
            }
        )

    df = pd.DataFrame(rows)
    for col in list(spec.numeric_features) + list(spec.categorical_features):
        if col not in df.columns:
            df[col] = np.nan
    return df
