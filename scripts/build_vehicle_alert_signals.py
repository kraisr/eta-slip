from __future__ import annotations

import argparse
from pathlib import Path
import time

import pandas as pd

from etaslip.gtfsrt.io import read_snapshot_gz
from etaslip.gtfsrt.parse import decode_feedmessage


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="raw/gtfsrt/feed=nyct%2Fgtfs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--route", action="append", default=["6"])
    ap.add_argument("--progress-every", type=int, default=5000)
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    paths = sorted(raw_root.rglob("*.pb.gz"))
    route_filter = set(args.route)
    rows: dict[tuple[int, str], dict] = {}
    started = time.time()

    for i, path in enumerate(paths, start=1):
        msg = decode_feedmessage(read_snapshot_gz(path))
        feed_ts = int(msg.header.timestamp or 0)
        if feed_ts <= 0:
            continue

        for ent in msg.entity:
            if ent.HasField("vehicle"):
                vehicle = ent.vehicle
                trip = vehicle.trip
                route_id = trip.route_id or ""
                if route_id not in route_filter:
                    continue
                trip_id = trip.trip_id or ""
                if not trip_id:
                    continue
                row = rows.setdefault((feed_ts, trip_id), _empty_row(feed_ts, trip_id))
                row["route_id"] = route_id
                row["vehicle_present"] = 1
                row["vehicle_timestamp"] = (
                    int(vehicle.timestamp) if vehicle.timestamp else None
                )
                row["vehicle_current_status"] = int(vehicle.current_status)
                row["vehicle_stop_id"] = vehicle.stop_id or ""
                row["vehicle_stop_sequence"] = (
                    int(vehicle.current_stop_sequence)
                    if vehicle.current_stop_sequence
                    else None
                )

            if ent.HasField("alert") and _alert_mentions_delay(ent.alert):
                for informed in ent.alert.informed_entity:
                    trip = informed.trip
                    route_id = trip.route_id or informed.route_id or ""
                    if route_id not in route_filter:
                        continue
                    trip_id = trip.trip_id or ""
                    if not trip_id:
                        continue
                    row = rows.setdefault(
                        (feed_ts, trip_id), _empty_row(feed_ts, trip_id)
                    )
                    row["route_id"] = route_id
                    row["trip_alert_delayed"] = 1

        if args.progress_every and i % args.progress_every == 0:
            print(
                {
                    "files": i,
                    "rows": len(rows),
                    "elapsed_sec": round(time.time() - started, 1),
                }
            )

    out = pd.DataFrame(rows.values())
    if out.empty:
        out = pd.DataFrame(columns=list(_empty_row(0, "").keys()))
    out["vehicle_movement_age_sec"] = out["feed_ts"] - pd.to_numeric(
        out["vehicle_timestamp"], errors="coerce"
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, compression="zstd", index=False)
    print(
        {
            "files": len(paths),
            "rows": len(out),
            "vehicle_rows": int(out["vehicle_present"].sum()) if not out.empty else 0,
            "alert_rows": int(out["trip_alert_delayed"].sum()) if not out.empty else 0,
            "out": str(out_path),
        }
    )


def _empty_row(feed_ts: int, trip_id: str) -> dict:
    return {
        "feed_ts": feed_ts,
        "next_trip_id": trip_id,
        "route_id": "",
        "vehicle_present": 0,
        "vehicle_timestamp": None,
        "vehicle_current_status": None,
        "vehicle_stop_id": "",
        "vehicle_stop_sequence": None,
        "trip_alert_delayed": 0,
    }


def _alert_mentions_delay(alert) -> bool:
    text = " ".join(t.text for t in alert.header_text.translation).lower()
    return "delay" in text or "delayed" in text


if __name__ == "__main__":
    main()
