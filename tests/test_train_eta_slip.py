from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from etaslip.modeling.dataset import DatasetSpec, load_gold, time_split
from etaslip.modeling.trainers import train_logreg, predict_proba


def write_gold_parquet(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal gold schema used by load_gold/train
    t = pa.table(
        {
            "feed_ts": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
            "match_status": pa.array(["matched"] * 6, pa.string()),
            "slip_ge_threshold": pa.array([0, 0, 1, 0, 1, 0], pa.int8()),
            "eta_t_minutes": pa.array([10, 12, 15, 8, 20, 9], pa.float32()),
            "top2_headway_sec": pa.array([300, 600, 900, 200, 1200, 250], pa.int32()),
            "num_arrivals_listed": pa.array([3, 2, 2, 4, 2, 3], pa.int16()),
            "dow": pa.array([0, 0, 0, 0, 0, 0], pa.int8()),
            "hour": pa.array([19, 19, 19, 19, 19, 19], pa.int8()),
            "minute": pa.array([0, 1, 2, 3, 4, 5], pa.int8()),
            "stop_id": pa.array(["A", "A", "B", "B", "A", "B"], pa.string()),
        }
    )
    pq.write_table(t, path, compression="zstd")


def test_load_and_train_logreg_smoke(tmp_path: Path):
    spec = DatasetSpec()
    f = tmp_path / "gold.parquet"
    write_gold_parquet(f)

    df = load_gold(f, spec)
    train_df, test_df = time_split(df, train_frac=0.67)

    pipe = train_logreg(train_df, spec)
    prob = predict_proba(pipe, test_df, spec)

    assert len(prob) == len(test_df)
    assert (prob >= 0).all() and (prob <= 1).all()
