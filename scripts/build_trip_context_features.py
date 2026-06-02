from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from etaslip.modeling.dataset import build_trip_context_from_events
from etaslip.pipelines.gold_eta_slip import list_silver_files


def load_stop_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    stop_ids = {
        line.strip()
        for line in Path(path).read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return stop_ids or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-root", default="silver")
    ap.add_argument("--source-feed", default="nyct%2Fgtfs")
    ap.add_argument("--dt", help="YYYY-MM-DD. If omitted, scans all silver dates.")
    ap.add_argument("--hour", help="HH. Requires --dt.")
    ap.add_argument("--route", action="append", default=["6"])
    ap.add_argument("--stop-ids-file")
    ap.add_argument("--raw-signal-path")
    ap.add_argument("--out", required=True)
    ap.add_argument("--progress-every", type=int, default=5000)
    args = ap.parse_args()

    paths = _list_files(
        Path(args.silver_root),
        source_feed=args.source_feed,
        dt=args.dt,
        hour=args.hour,
    )
    route_filter = set(args.route)
    stop_filter = load_stop_filter(args.stop_ids_file)
    signal_groups = _load_signal_groups(args.raw_signal_path, route_filter)

    parts: list[pd.DataFrame] = []
    started = time.time()
    for i, path in enumerate(paths, start=1):
        events = pd.read_parquet(
            path, columns=["feed_ts", "route_id", "trip_id", "stop_id", "eta"]
        )
        events = events[events["route_id"].isin(route_filter)].copy()
        if events.empty:
            continue

        feed_ts = int(events["feed_ts"].iloc[0])
        context = build_trip_context_from_events(
            events,
            vehicle_alert_signals=signal_groups.get(feed_ts),
        )
        if stop_filter:
            context = context[context["stop_id"].isin(stop_filter)]
        if not context.empty:
            parts.append(context)

        if args.progress_every and i % args.progress_every == 0:
            print(
                {
                    "files": i,
                    "parts": len(parts),
                    "elapsed_sec": round(time.time() - started, 1),
                }
            )

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not out.empty:
        out = out.drop_duplicates(["feed_ts", "next_trip_id", "stop_id"], keep="last")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, compression="zstd", index=False)
    print(
        {
            "files": len(paths),
            "rows": len(out),
            "out": str(out_path),
            "elapsed_sec": round(time.time() - started, 1),
        }
    )


def _list_files(
    silver_root: Path,
    *,
    source_feed: str,
    dt: str | None,
    hour: str | None,
) -> list[Path]:
    if dt:
        return list_silver_files(silver_root, source_feed=source_feed, dt=dt, hour=hour)
    base = silver_root / "trip_updates" / f"feed={source_feed}"
    return sorted(base.glob("dt=*/hour=*/*.parquet"))


def _load_signal_groups(
    path: str | None,
    route_filter: set[str],
) -> dict[int, pd.DataFrame]:
    if not path:
        return {}
    signals = pd.read_parquet(path)
    if "route_id" in signals.columns:
        signals = signals[signals["route_id"].isin(route_filter)].copy()
    return {
        int(feed_ts): group.copy()
        for feed_ts, group in signals.groupby("feed_ts", sort=False)
    }


if __name__ == "__main__":
    main()
