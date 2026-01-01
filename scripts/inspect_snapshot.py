import argparse
from pathlib import Path

from etaslip.gtfsrt import format_summary, summarize_snapshot


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to a .pb.gz snapshot")
    p.add_argument("--route", default="6")
    p.add_argument("--top-n", type=int, default=20)
    args = p.parse_args()

    summary = summarize_snapshot(Path(args.path), route_id=args.route)
    print(format_summary(summary, route_id=args.route, top_n=args.top_n))


if __name__ == "__main__":
    main()
