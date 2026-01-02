from __future__ import annotations

import argparse

import pandas as pd

from etaslip.modeling.dataset import DatasetSpec, load_gold


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--gold-path",
        required=True,
        help="Gold parquet file OR directory (will read all *.parquet)",
    )
    args = ap.parse_args()

    spec = DatasetSpec()
    df = load_gold(args.gold_path, spec)

    print(f"Rows (matched only): {len(df)}")
    print(f"Pos rate: {df[spec.target_col].mean():.4f}")
    if "slip_seconds" in df.columns:
        s = pd.to_numeric(df["slip_seconds"], errors="coerce").dropna()
        if len(s):
            print("Slip seconds quantiles:")
            print(s.quantile([0.5, 0.75, 0.9, 0.95, 0.99]).to_string())

    print("\nPos rate by stop_id (top 20 by rows):")
    g = (
        df.groupby("stop_id")[spec.target_col]
        .agg(["count", "mean"])
        .sort_values("count", ascending=False)
        .head(20)
    )
    print(g.to_string())

    print("\nFeature null rates:")
    for c in list(spec.numeric_features) + list(spec.categorical_features):
        if c in df.columns:
            print(f"  {c:<20} {df[c].isna().mean():.4f}")


if __name__ == "__main__":
    main()
