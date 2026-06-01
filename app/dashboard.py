#!/usr/bin/env python3
from __future__ import annotations

import time
from datetime import datetime

import joblib
import numpy as np
import streamlit as st
from streamlit_extras.st_keyup import st_keyup
from streamlit_autorefresh import st_autorefresh

from etaslip.online.constants import (
    DEFAULT_FEED_URL,
    NY_TZ,
    SENSITIVITIES,
    STOP_IDS_FILE_DEFAULT,
)
from etaslip.online.features import (
    build_features_from_events,
    build_service_issue_summary,
    build_unscored_arrival_fallback,
)
from etaslip.online.gtfsrt import (
    apply_filters_with_route_fallback,
    fetch_feed_bytes,
    maybe_gunzip,
    parse_tripupdates,
)
from etaslip.online.model_artifacts import (
    load_dataset_spec,
    load_serving_params,
    load_threshold,
    resolve_model_dir_latest,
)
from etaslip.online.stops import load_stop_id_to_name_map, read_ordered_stop_ids

from ui_utils import bar_html

# from perf_panel import render_precision_panel


def main() -> None:
    st.set_page_config(page_title="ETA Slip", page_icon="🚇", layout="wide")

    st.title("🚇 Southbound 6 Train Delay Risk")
    st.caption(
        "Real-time: which Manhattan southbound stations are most likely to see an ETA slip soon."
    )

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

        top_n = st.slider(
            "Number of stations to show",
            min_value=1,
            max_value=max_n,
            value=min(12, max_n),
        )
        st.button("Refresh now")  # triggers rerun

    # Auto-refresh every 30s
    # tick = st_autorefresh(interval=30_000, key="auto_refresh")
    st_autorefresh(interval=30_000, key="auto_refresh")
    # st.caption(f"refresh tick: {tick}")

    # --- Load model (auto latest) ---
    model_dir = resolve_model_dir_latest()
    model_path = model_dir / f"model_{model_name}.joblib"
    if not model_path.exists():
        st.error(f"Model not found: {model_path}")
        st.stop()

    spec = load_dataset_spec(model_dir)

    @st.cache_resource
    def _load_model(p: str):
        return joblib.load(p)

    pipe = _load_model(str(model_path))

    base_thr = load_threshold(model_dir, model_name) or 0.5
    serving_params = load_serving_params(model_dir)
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
    events, route_filter_used, used_route_fallback = apply_filters_with_route_fallback(
        events,
        primary_route_filter={"6"},
        fallback_route_filter={"6X"},
        stop_ids=stop_ids_set,
        stop_suffix="S",
    )
    service_issues = build_service_issue_summary(events, stop_name_map=name_map)

    feats = build_features_from_events(
        events,
        spec=spec,
        arrival_rank=serving_params.arrival_rank,
        min_lead_sec=serving_params.min_lead_sec,
        history_state=st.session_state.setdefault("feature_history_state", {}),
    )

    if feats.empty:
        _render_service_issue_notice(service_issues)
        fallback = build_unscored_arrival_fallback(
            events,
            stop_order=stop_order,
            stop_name_map=name_map,
            min_lead_sec=serving_params.min_lead_sec,
        )
        if fallback.empty:
            st.warning(
                "No covered southbound 6 arrivals are present in the live feed right now."
            )
        else:
            st.warning(
                "No arrivals meet the model scoring constraints right now. "
                "Showing unscored live arrivals instead."
            )
            st.caption(
                "Scoring needs at least two future arrivals at a station and one "
                f"arrival at least {serving_params.min_lead_sec // 60} minutes away."
            )
            st.dataframe(fallback, hide_index=True, use_container_width=True)
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
    colC.metric("Alert threshold", f"{thr * 100:.1f}%")

    _render_route_fallback_notice(
        used_route_fallback=used_route_fallback,
        route_filter_used=route_filter_used,
    )
    _render_service_issue_notice(service_issues)

    st.divider()

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
                    Train ETA: <b>{eta_min:.0f} min</b> &nbsp;•&nbsp; Delay risk: <b>{pct:.0f}%</b>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.subheader("✅ No alerts right now")
        st.write(
            "Everything looks normal for the covered Manhattan southbound stops (based on the current model)."
        )

    st.divider()

    # --- Line strip with search ---
    st.subheader("Line strip")

    q = st_keyup(
        "Search station",
        placeholder="Type a station name (e.g., Grand Central)",
        key="station_search",
    )

    filtered = out.copy()
    if q.strip():
        filtered = filtered[
            filtered["station_name"]
            .astype(str)
            .str.contains(q.strip(), case=False, na=False)
        ]

    # Pick Top-N by risk among the filtered set, then display in travel order
    top = filtered.sort_values("proba_slip", ascending=False).head(int(top_n)).copy()
    top = top.sort_values("order", ascending=False).reset_index(drop=True)

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
                st.caption(f"Risk: {pct:.0f}%" if pct >= 0.5 else "Risk: <1%")
            with right:
                st.markdown(f"**Train ETA**\n\n{eta_min:.0f} min")

    # # Download data
    # with st.expander("Download data"):
    #     csv = (
    #         out.sort_values("proba_slip", ascending=False)
    #         .to_csv(index=False)
    #         .encode("utf-8")
    #     )
    #     st.download_button(
    #         "Download CSV", data=csv, file_name="eta_slip_scores.csv", mime="text/csv"
    #     )

    # st.divider()

    # render_precision_panel(
    #     spec=spec,
    #     pipe=pipe,
    #     model_name=model_name,
    #     thr=thr,
    #     gold_root="gold/eta_slip",
    #     max_days=14,
    # )


def _render_service_issue_notice(service_issues) -> None:
    if service_issues.empty:
        return

    st.warning(
        "GTFS-RT marks some covered stop updates as skipped, no-data, or otherwise non-standard."
    )
    with st.expander("Service data details"):
        st.dataframe(service_issues, hide_index=True, use_container_width=True)


def _render_route_fallback_notice(
    *, used_route_fallback: bool, route_filter_used: set[str]
) -> None:
    if not used_route_fallback:
        return
    routes = ", ".join(sorted(route_filter_used))
    st.info(
        f"No regular southbound 6 updates are present for the covered stops right now. "
        f"Showing route {routes} updates from the live feed."
    )


if __name__ == "__main__":
    main()
