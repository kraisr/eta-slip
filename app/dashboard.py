#!/usr/bin/env python3
from __future__ import annotations

import time
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
from streamlit_extras.st_keyup import st_keyup

from etaslip.modeling.dataset import DatasetSpec
from etaslip.online.constants import (
    DEFAULT_FEED_URL,
    NY_TZ,
    SENSITIVITIES,
    STOP_IDS_FILE_DEFAULT,
)
from etaslip.online.features import build_features_from_events
from etaslip.online.gtfsrt import apply_filters, fetch_feed_bytes, maybe_gunzip, parse_tripupdates
from etaslip.online.model_artifacts import load_threshold, resolve_model_dir_latest
from etaslip.online.stops import load_stop_id_to_name_map, read_ordered_stop_ids

from ui_utils import bar_html


def main() -> None:
    st.set_page_config(page_title="ETA Slip", page_icon="🚇", layout="wide")

    st.title("🚇 Southbound 6 Train Delay Risk")
    st.caption("Real-time: which Manhattan southbound stations are most likely to see an ETA slip soon.")

    # --- Sidebar ---
    with st.sidebar:
        st.header("Controls")
        model_name = st.selectbox("Model", ["xgb", "logreg"], index=0)
        sensitivity_label = st.selectbox(
            "Alert sensitivity",
            [s.label for s in SENSITIVITIES],
            index=1,
            help="Higher sensitivity = more alerts.",
        )

        # Keep these hidden from normal users (not widgets)
        stop_ids_path = STOP_IDS_FILE_DEFAULT

        stop_order = read_ordered_stop_ids(stop_ids_path)
        max_n = len(stop_order)

        top_n = st.slider("Number of stations to show", min_value=1, max_value=max_n, value=min(12, max_n))
        st.button("Refresh now")  # triggers rerun

    # Auto-refresh every 30 seconds if available in your Streamlit version
    if callable(getattr(st, "autorefresh", None)):
        st.autorefresh(interval=30_000, key="auto_refresh_v2")

    # --- Load model (auto latest) ---
    model_dir = resolve_model_dir_latest()
    model_path = model_dir / f"model_{model_name}.joblib"
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    spec = DatasetSpec()

    @st.cache_resource
    def _load_model(p: str):
        return joblib.load(p)

    pipe = _load_model(str(model_path))

    base_thr = load_threshold(model_dir, model_name) or 0.5
    sens = next(s for s in SENSITIVITIES if s.label == sensitivity_label)
    thr = float(np.clip(base_thr * sens.multiplier, 0.001, 0.99))  # 0.1% floor

    # --- Station names from JSON mapping ---
    name_map = load_stop_id_to_name_map()

    # --- Fetch + score live ---
    try:
        raw = fetch_feed_bytes(DEFAULT_FEED_URL)
        feed_bytes = maybe_gunzip(raw)
        events = parse_tripupdates(feed_bytes)
    except Exception as e:
        st.error(f"Live feed fetch/parse failed: {e}")
        st.stop()

    stop_ids_set = set(stop_order)
    events = apply_filters(
        events,
        route_filter={"6"},
        stop_ids=stop_ids_set,
        stop_suffix="S",
    )

    feats = build_features_from_events(
        events,
        spec=spec,
        arrival_rank=3,
        min_lead_sec=480,
    )

    if feats.empty:
        st.warning("No arrivals met the constraints right now (after filters). Try again in a minute.")
        st.stop()

    X = feats[list(spec.numeric_features) + list(spec.categorical_features)]
    proba = pipe.predict_proba(X)[:, 1]

    out = feats.copy()
    out["proba_slip"] = proba
    out["pred_slip"] = out["proba_slip"] >= thr

    out["station_name"] = out["stop_id"].map(name_map).fillna(out["stop_id"])

    order_index = {sid: i for i, sid in enumerate(stop_order)}
    out["order"] = out["stop_id"].map(order_index).astype("Int64")

    # Header metrics (more compact updated + threshold shown as %)
    feed_ts = int(events["feed_ts"].iloc[0]) if not events.empty else int(time.time())
    updated = datetime.fromtimestamp(feed_ts, tz=NY_TZ).strftime("%H:%M %Z")

    n_alerts = int(out["pred_slip"].sum())

    colA, colB, colC = st.columns([1.3, 1.0, 1.0])
    colA.metric("Updated", updated)
    colB.metric("Alerts now", str(n_alerts))
    colC.metric("Alert threshold", f"{thr*100:.1f}%")

    # --- Alerts ---
    alerts = out[out["pred_slip"]].copy().sort_values("proba_slip", ascending=False)
    if len(alerts) > 0:
        st.subheader("⚠️ Alerts")
        for _, r in alerts.head(5).iterrows():
            pct = float(r["proba_slip"]) * 100.0
            eta_min = float(r["eta_t_minutes"])
            station = str(r["station_name"])
            st.markdown(
                f"""
                <div style="
                    padding: 14px 16px;
                    border-radius: 14px;
                    background: rgba(215, 48, 39, 0.14);
                    border: 1px solid rgba(215, 48, 39, 0.25);
                    margin-bottom: 10px;">
                  <div style="font-size: 18px; font-weight: 800;">{station}</div>
                  <div style="font-size: 14px; opacity: 0.95;">
                    Next train ETA: <b>{eta_min:.0f} min</b> &nbsp;•&nbsp; Delay risk: <b>{pct:.0f}%</b>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.subheader("✅ No alerts right now")
        st.write("Everything looks normal for the covered Manhattan southbound stops (based on the current model).")

    st.divider()

    # --- Line strip with search ---
    st.subheader("Line strip")

    q = st_keyup("Search station", placeholder="Type a station name (e.g., Grand Central)", key="station_search")

    filtered = out.copy()
    if q.strip():
        filtered = filtered[filtered["station_name"].astype(str).str.contains(q.strip(), case=False, na=False)]

    # Pick Top-N by risk among the filtered set, then display in travel order
    top = filtered.sort_values("proba_slip", ascending=False).head(int(top_n)).copy()
    top = top.sort_values("order", ascending=True).reset_index(drop=True)

    if top.empty:
        st.info("No stations match your search.")
    else:
        for _, r in top.iterrows():
            station = str(r["station_name"])
            p = float(r["proba_slip"])
            eta_min = float(r["eta_t_minutes"])
            pct = p * 100.0

            left, mid, right = st.columns([2.6, 2.2, 1.2])
            with left:
                st.markdown(f"**{station}**")
            with mid:
                st.markdown(bar_html(p, thr), unsafe_allow_html=True)
                st.caption(f"Risk: {pct:.0f}%")
            with right:
                st.markdown(f"**ETA**\n\n{eta_min:.0f} min")

    # Download data
    with st.expander("Download data"):
        csv = out.sort_values("proba_slip", ascending=False).to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="eta_slip_scores.csv", mime="text/csv")


if __name__ == "__main__":
    main()
