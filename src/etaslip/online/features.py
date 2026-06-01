from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from etaslip.modeling.dataset import (
    HISTORY_NUMERIC_FEATURES,
    DatasetSpec,
    add_derived_features,
    needs_history_features,
)
from etaslip.online.constants import NY_TZ

NORMAL_RELATIONSHIPS = {"", "SCHEDULED"}


def build_features_from_events(
    events: pd.DataFrame,
    *,
    spec: DatasetSpec,
    arrival_rank: int,
    min_lead_sec: int,
    history_state: dict | None = None,
    tz: ZoneInfo = NY_TZ,
) -> pd.DataFrame:
    """
    One row per stop_id:
      eta_t_minutes, top2_headway_sec, num_arrivals_listed, dow, hour, minute, stop_id

    The selected arrival obeys min_lead_sec/arrival_rank, while headway and
    num_arrivals_listed are computed from all future arrivals to match gold.
    """
    if events.empty:
        return pd.DataFrame(
            columns=list(spec.numeric_features) + list(spec.categorical_features)
        )

    feed_ts = int(events["feed_ts"].iloc[0])
    feed_dt = datetime.fromtimestamp(feed_ts, tz=tz)

    e = events.copy()
    if "trip_id" not in e.columns:
        e["trip_id"] = ""
    e["eta"] = pd.to_numeric(e["eta"], errors="coerce")
    e = e[e["eta"].notna()].copy()
    e["lead_sec"] = e["eta"].astype(int) - int(feed_ts)

    e = e[e["lead_sec"] > 0]
    if e.empty:
        return pd.DataFrame(
            columns=list(spec.numeric_features) + list(spec.categorical_features)
        )

    use_history = needs_history_features(spec)
    if use_history:
        e = _add_online_history_features(e, history_state)

    rows: list[dict] = []
    for stop_id, g in e.groupby("stop_id", sort=False):
        g = g.sort_values("lead_sec").copy()
        leads = g["lead_sec"].to_numpy(dtype=float)
        num = int(len(leads))
        if num < 2:
            continue

        eligible = g
        if min_lead_sec > 0:
            eligible = eligible[eligible["lead_sec"] >= int(min_lead_sec)]

        if len(eligible) < arrival_rank:
            continue

        selected = eligible.iloc[arrival_rank - 1]
        lead_k = float(selected["lead_sec"])
        headway = float(leads[1] - leads[0])
        row = {
            "stop_id": str(stop_id),
            "eta_t_minutes": lead_k / 60.0,
            "top2_headway_sec": headway,
            "num_arrivals_listed": num,
            "dow": int(feed_dt.weekday()),
            "hour": int(feed_dt.hour),
            "minute": int(feed_dt.minute),
        }
        if use_history:
            for col in HISTORY_NUMERIC_FEATURES:
                row[col] = float(selected.get(col, _history_default(col)))
        rows.append(row)

    if use_history:
        _update_online_history_state(e, history_state)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = add_derived_features(df)
    for col in list(spec.numeric_features) + list(spec.categorical_features):
        if col not in df.columns:
            df[col] = np.nan
    return df


def build_unscored_arrival_fallback(
    events: pd.DataFrame,
    *,
    stop_order: list[str],
    stop_name_map: dict[str, str],
    min_lead_sec: int,
    max_rows: int = 12,
) -> pd.DataFrame:
    """
    Build a non-model fallback table from live feed rows when model features are empty.
    """
    columns = [
        "station",
        "stop_id",
        "next_eta_min",
        "future_arrivals",
        "model_eligible_arrivals",
        "headway_min",
        "service_status",
    ]
    if events.empty or "eta" not in events.columns:
        return pd.DataFrame(columns=columns)

    feed_ts = int(events["feed_ts"].iloc[0])
    status_by_stop = {
        str(stop_id): _service_status_label(group)
        for stop_id, group in events.groupby("stop_id", sort=False)
    }
    df = events.copy()
    df["eta"] = pd.to_numeric(df["eta"], errors="coerce")
    df = df[df["eta"].notna()].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    df["lead_sec"] = df["eta"].astype(int) - feed_ts
    df = df[df["lead_sec"] > 0].copy()
    if df.empty:
        return pd.DataFrame(columns=columns)

    order_index = {sid: i for i, sid in enumerate(stop_order)}
    rows: list[dict] = []
    for stop_id, group in df.groupby("stop_id", sort=False):
        g = group.sort_values("lead_sec")
        leads = g["lead_sec"].to_numpy(dtype=float)
        eligible = leads[leads >= int(min_lead_sec)]
        headway = (leads[1] - leads[0]) / 60.0 if len(leads) >= 2 else np.nan
        rows.append(
            {
                "station": stop_name_map.get(str(stop_id), str(stop_id)),
                "stop_id": str(stop_id),
                "next_eta_min": round(float(leads[0] / 60.0), 1),
                "future_arrivals": int(len(leads)),
                "model_eligible_arrivals": int(len(eligible)),
                "headway_min": round(float(headway), 1)
                if np.isfinite(headway)
                else np.nan,
                "service_status": status_by_stop.get(str(stop_id), "Normal"),
                "order": order_index.get(str(stop_id), 9999),
            }
        )

    out = pd.DataFrame(rows).sort_values("order").drop(columns=["order"])
    return out.head(int(max_rows)).reset_index(drop=True)


def build_service_issue_summary(
    events: pd.DataFrame,
    *,
    stop_name_map: dict[str, str],
    max_rows: int = 8,
) -> pd.DataFrame:
    columns = ["station", "stop_id", "service_status", "updates"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    issues = events[_nonstandard_relationship_mask(events)].copy()
    if issues.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for stop_id, group in issues.groupby("stop_id", sort=True):
        rows.append(
            {
                "station": stop_name_map.get(str(stop_id), str(stop_id)),
                "stop_id": str(stop_id),
                "service_status": _service_status_label(group),
                "updates": int(len(group)),
            }
        )
    return pd.DataFrame(rows).head(int(max_rows))


def _add_online_history_features(
    events: pd.DataFrame, history_state: dict | None
) -> pd.DataFrame:
    out = events.copy()
    for col in HISTORY_NUMERIC_FEATURES:
        out[col] = _history_default(col)

    if history_state is None:
        return _add_station_history_features(out)

    last = history_state.setdefault("last", {})
    eta_delta_hist = history_state.setdefault("eta_delta_hist", {})
    lead_delta_hist = history_state.setdefault("lead_delta_hist", {})

    for idx, row in out.iterrows():
        key = _history_key(row["stop_id"], row.get("trip_id", ""))
        prev = last.get(key)
        if not prev:
            continue

        gap = float(row["feed_ts"] - prev["feed_ts"])
        if gap <= 0:
            continue

        lead_sec = float(row["lead_sec"])
        eta_delta = float(row["eta"] - prev["eta"])
        lead_delta = float(lead_sec - prev["lead_sec"])
        eta_window = [*eta_delta_hist.get(key, []), eta_delta][-3:]
        lead_window = [*lead_delta_hist.get(key, []), lead_delta][-3:]

        out.at[idx, "has_trip_history"] = 1.0
        out.at[idx, "fresh_trip_history_10m"] = float(gap <= 600)
        out.at[idx, "history_gap_sec"] = min(gap, 9999.0)
        out.at[idx, "eta_abs_delta_prev_sec"] = eta_delta
        out.at[idx, "lead_delta_prev_sec"] = lead_delta
        out.at[idx, "lead_delta_expected_slip_sec"] = lead_delta + gap
        out.at[idx, "eta_abs_delta_mean3_sec"] = float(np.mean(eta_window))
        out.at[idx, "eta_abs_delta_max3_sec"] = float(np.max(eta_window))
        out.at[idx, "lead_delta_mean3_sec"] = float(np.mean(lead_window))
        out.at[idx, "lead_delta_max3_sec"] = float(np.max(lead_window))

    return _add_station_history_features(out)


def _add_station_history_features(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    by_stop = out.groupby("stop_id", sort=False)
    out["station_eta_abs_delta_mean_sec"] = by_stop["eta_abs_delta_prev_sec"].transform(
        "mean"
    )
    out["station_eta_abs_delta_max_sec"] = by_stop["eta_abs_delta_prev_sec"].transform(
        "max"
    )
    out["station_lead_delta_mean_sec"] = by_stop["lead_delta_prev_sec"].transform(
        "mean"
    )
    out["station_lead_delta_max_sec"] = by_stop["lead_delta_prev_sec"].transform("max")
    out["station_share_positive_eta_delta"] = by_stop[
        "eta_abs_delta_prev_sec"
    ].transform(lambda s: (s > 0).mean())
    return out


def _update_online_history_state(
    events: pd.DataFrame, history_state: dict | None
) -> None:
    if history_state is None:
        return

    last = history_state.setdefault("last", {})
    eta_delta_hist = history_state.setdefault("eta_delta_hist", {})
    lead_delta_hist = history_state.setdefault("lead_delta_hist", {})

    for row in events.itertuples(index=False):
        key = _history_key(row.stop_id, getattr(row, "trip_id", ""))
        if int(getattr(row, "has_trip_history", 0)) == 1:
            eta_delta_hist[key] = [
                *eta_delta_hist.get(key, []),
                float(getattr(row, "eta_abs_delta_prev_sec", 0.0)),
            ][-2:]
            lead_delta_hist[key] = [
                *lead_delta_hist.get(key, []),
                float(getattr(row, "lead_delta_prev_sec", 0.0)),
            ][-2:]
        last[key] = {
            "feed_ts": float(row.feed_ts),
            "eta": float(row.eta),
            "lead_sec": float(row.lead_sec),
        }


def _history_key(stop_id: object, trip_id: object) -> str:
    return f"{stop_id}||{trip_id}"


def _history_default(col: str) -> float:
    return 9999.0 if col == "history_gap_sec" else 0.0


def _nonstandard_relationship_mask(events: pd.DataFrame) -> pd.Series:
    mask = pd.Series(False, index=events.index)
    for col in ("trip_schedule_relationship", "stop_schedule_relationship"):
        if col not in events.columns:
            continue
        rel = events[col].fillna("").astype(str)
        mask = mask | ~rel.isin(NORMAL_RELATIONSHIPS)
    return mask


def _service_status_label(events: pd.DataFrame) -> str:
    statuses: set[str] = set()
    for col in ("trip_schedule_relationship", "stop_schedule_relationship"):
        if col not in events.columns:
            continue
        statuses.update(
            rel
            for rel in events[col].fillna("").astype(str).unique()
            if rel not in NORMAL_RELATIONSHIPS
        )
    return ", ".join(sorted(statuses)) if statuses else "Normal"
