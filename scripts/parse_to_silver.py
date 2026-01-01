from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from etaslip.pipelines.silver_tripupdates import parse_raw_snapshot_to_silver


def iter_raw_files(raw_root: Path, source_feed: str, dt: Optional[str], hour: Optional[str]):
    base = raw_root / f"feed={source_feed}"
    if dt is not None and hour is not None:
        pattern = base / f"dt={dt}" / f"hour={hour}" / "*.pb.gz"
    elif dt is not None:
        pattern = base / f"dt={dt}" / "hour=*" / "*.pb.gz"
    else:
        pattern = base / "dt=*" / "hour=*" / "*.pb.gz"

    for p in sorted(pattern.parent.parent.parent.glob(str(pattern).split(str(base))[1].lstrip("/"))):
        # ^ a bit awkward to avoid Path.glob limitations across absolute strings
        # We'll just yield later using base.glob with a relative pattern
        yield p


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", default="raw/gtfsrt")
    p.add_argument("--silver-root", default="silver")
    p.add_argument("--source-feed", default="nyct%2Fgtfs")
    p.add_argument("--dt", help="YYYY-MM-DD (optional)")
    p.add_argument("--hour", help="HH (00-23) (optional; requires --dt)")
    p.add_argument("--route", action="append", default=[], help="route_id filter (repeatable), e.g. --route 6")
    p.add_argument("--include-all-routes", action="store_true", help="ignore --route filters")
    p.add_argument("--no-skip", action="store_true", help="re-write outputs even if they exist")
    args = p.parse_args()

    raw_root = Path(args.raw_root)
    silver_root = Path(args.silver_root)

    if args.hour and not args.dt:
        raise SystemExit("--hour requires --dt")

    route_filter = None
    if not args.include_all_routes and args.route:
        route_filter = set(args.route)

    # Safer globbing: use base.glob with relative patterns
    base = raw_root / f"feed={args.source_feed}"
    if args.dt and args.hour:
        rel = f"dt={args.dt}/hour={args.hour}/*.pb.gz"
    elif args.dt:
        rel = f"dt={args.dt}/hour=*/*.pb.gz"
    else:
        rel = "dt=*/hour=*/*.pb.gz"

    raw_files = sorted(base.glob(rel))

    wrote = 0
    skipped_or_empty = 0
    for f in raw_files:
        out = parse_raw_snapshot_to_silver(
            f,
            silver_root=silver_root,
            source_feed=args.source_feed,
            route_filter=route_filter,
            skip_if_exists=not args.no_skip,
        )
        if out is None:
            skipped_or_empty += 1
        else:
            wrote += 1

    print(f"Raw files scanned: {len(raw_files)}")
    print(f"Parquet parts written: {wrote}")
    print(f"Skipped/empty: {skipped_or_empty}")
    print(f"Silver root: {silver_root.resolve()}")


if __name__ == "__main__":
    main()
