from __future__ import annotations

import re
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from etaslip.modeling.dataset import DatasetSpec

NY_TZ = ZoneInfo("America/New_York")

_DT_RE = re.compile(r"/dt=(\d{4}-\d{2}-\d{2})/")


def _list_gold_files(gold_root: str = "gold/eta_slip", max_days: int = 14) -> list[Path]:
    """
    Auto-discovers gold parquet files and keeps the most recent N distinct dt=YYYY-MM-DD partitions.
    """
    root = Path(gold_root)
    if not root.exists():
        return []

    files = sorted(root.glob("**/*.parquet"))
    if not files:
        return []

    items: list[tuple[str, Path]] = []
    for f in files:
        m = _DT_RE.search(str(f).replace("\\", "/"))
        if m:
            items.append((m.group(1), f))

    if not items:
        # fallback: just return most recent few files
        return sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)[:50]

    dts_sorted = sorted({dt for dt, _ in items})
    keep_dts = set(dts_sorted[-max_days:])
    keep_files = [f for dt, f in items if dt in keep_dts]
    return sorted(keep_files)


@st.cache_data(ttl=300, show_spinner=False)
def _load_gold(files: tuple[str, ...]) -> pd.DataFrame:
    """
    Loads & concatenates gold parquet files.
    Cached for dashboard refresh.
    """
    if not files:
        return pd.DataFrame()

    dfs = [pd.read_parquet(p) for p in files]
    return pd.concat(dfs, ignore_index=True)


def _eval_precision_by_day(
    df: pd.DataFrame,
    *,
    pipe,
    spec: DatasetSpec,
    thr: float,
    model_name: str,
) -> pd.DataFrame:
    """
    Daily precision for alerts: TP / (TP + FP), where alert = proba >= thr.
    """
    if df.empty:
        return pd.DataFrame()

    # Match filter if present in gold
    if spec.match_col in df.columns:
        df = df[df[spec.match_col] == spec.required_match_value].copy()

    needed = set([spec.target_col, "feed_ts"]) | set(spec.numeric_features) | set(spec.categorical_features)
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return pd.DataFrame()

    # Day bucket (NY time)
    dt = pd.to_datetime(df["feed_ts"], unit="s", utc=True).dt.tz_convert(NY_TZ)
    df["day"] = dt.dt.date.astype(str)

    X = df[list(spec.numeric_features) + list(spec.categorical_features)]
    y = df[spec.target_col].astype(int).to_numpy()

    prob = pipe.predict_proba(X)[:, 1]
    pred = prob >= float(thr)

    df["y"] = y
    df["pred"] = pred

    rows: list[dict] = []
    for day, g in df.groupby("day", sort=True):
        y_d = g["y"].to_numpy(dtype=int)
        p_d = g["pred"].to_numpy(dtype=bool)

        tp = int(np.sum((p_d == 1) & (y_d == 1)))
        fp = int(np.sum((p_d == 1) & (y_d == 0)))

        # If no alerts that day, precision is NaN (chart will show a gap)
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else np.nan

        rows.append(
            {
                "day": day,
                "model": model_name,
                "precision": prec,
                "precision_pct": prec * 100.0 if np.isfinite(prec) else np.nan,
                "alerts": int(np.sum(p_d)),
                "threshold": float(thr),
                "n": int(len(g)),
            }
        )

    return pd.DataFrame(rows)


def render_precision_panel(
    *,
    spec: DatasetSpec,
    pipe,
    model_name: str,
    thr: float,
    gold_root: str = "gold/eta_slip",
    max_days: int = 14,
) -> None:
    """
    Bottom-of-page panel: precision-only line chart.
    """
    import altair as alt

    files = _list_gold_files(gold_root=gold_root, max_days=max_days)
    if not files:
        st.info("No gold files found yet to compute precision stats.")
        return

    df_gold = _load_gold(tuple(str(p) for p in files))
    if df_gold.empty:
        st.info("Gold files were found, but no rows were loaded.")
        return

    mdf = _eval_precision_by_day(df_gold, pipe=pipe, spec=spec, thr=thr, model_name=model_name)
    if mdf.empty:
        st.info("Could not compute precision (missing columns or no evaluable rows).")
        return

    st.subheader("Recent accuracy (precision)")
    st.caption(f"From last {max_days} days.")

    chart = (
        alt.Chart(mdf)
        .mark_line(point=True)
        .encode(
            x=alt.X("day:N", title="Day"),
            y=alt.Y("precision_pct:Q", title="Alert precision (%)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["day", "model", "precision_pct", "alerts", "threshold", "n"],
        )
    )

    st.altair_chart(chart, use_container_width=True)
