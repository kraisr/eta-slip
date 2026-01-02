from __future__ import annotations

import argparse
from pathlib import Path

from etaslip.pipelines.gold_eta_slip import (
    build_gold_eta_slip_for_files,
    gold_output_path,
    list_silver_files,
    write_gold_table,
)


def load_stop_filter(path: str | None) -> set[str] | None:
    if not path:
        return None
    p = Path(path)
    stop_ids = set()
    for line in p.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            stop_ids.add(s)
    return stop_ids or None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--silver-root", default="silver")
    ap.add_argument("--gold-root", default="gold")
    ap.add_argument("--source-feed", default="nyct%2Fgtfs")
    ap.add_argument("--dt", required=True, help="YYYY-MM-DD")
    ap.add_argument(
        "--hour", help="HH (00-23). If omitted, builds for all hours in dt."
    )
    ap.add_argument(
        "--route",
        action="append",
        default=[],
        help="route_id filter (repeatable), e.g. --route 6",
    )
    ap.add_argument(
        "--include-all-routes", action="store_true", help="ignore --route filters"
    )
    ap.add_argument(
        "--stop-ids-file", help="Optional text file with stop_id per line (watchlist)."
    )
    ap.add_argument("--horizon-sec", type=int, default=300)
    ap.add_argument("--tolerance-sec", type=int, default=120)
    ap.add_argument("--slip-threshold-sec", type=int, default=120)
    ap.add_argument(
        "--no-skip", action="store_true", help="overwrite gold output if exists"
    )
    ap.add_argument(
        "--min-lead-sec",
        type=int,
        default=0,
        help="require eta_t >= feed_ts + min_lead_sec",
    )
    ap.add_argument(
        "--arrival-rank",
        type=int,
        default=1,
        help="use Nth upcoming arrival at t (1=next)",
    )

    args = ap.parse_args()

    silver_root = Path(args.silver_root)
    gold_root = Path(args.gold_root)

    route_filter = None
    if not args.include_all_routes and args.route:
        route_filter = set(args.route)

    stop_filter = load_stop_filter(args.stop_ids_file)

    hours = [args.hour] if args.hour is not None else [None]

    for hr in hours:
        files = list_silver_files(
            silver_root, source_feed=args.source_feed, dt=args.dt, hour=hr
        )
        if not files:
            print(f"No silver files found for dt={args.dt}, hour={hr}")
            continue

        table, stats = build_gold_eta_slip_for_files(
            files,
            horizon_sec=args.horizon_sec,
            tolerance_sec=args.tolerance_sec,
            slip_threshold_sec=args.slip_threshold_sec,
            route_filter=route_filter,
            stop_filter=stop_filter,
            min_lead_sec=args.min_lead_sec,
            arrival_rank=args.arrival_rank,
        )

        out_path = gold_output_path(
            gold_root,
            dt=args.dt,
            hour=hr,
            horizon_sec=args.horizon_sec,
            source_feed=args.source_feed,
        )

        if out_path.exists() and not args.no_skip:
            print(f"Skip existing: {out_path}")
            continue

        write_gold_table(table, out_path)

        # quick stats
        pos = 0
        if table.num_rows > 0 and "slip_ge_threshold" in table.column_names:
            col = table["slip_ge_threshold"].to_pylist()
            pos = sum(1 for v in col if v == 1)

        print(f"Wrote: {out_path}")
        print(
            f"  snapshots={stats.snapshots}  examples={stats.candidate_examples}  "
            f"matched={stats.matched}  missing_at_tplus={stats.missing_at_tplus}  "
            f"match_rate={stats.match_rate:.3f}  pos_rate={(pos / stats.candidate_examples) if stats.candidate_examples else 0.0:.3f}"
        )


if __name__ == "__main__":
    main()
