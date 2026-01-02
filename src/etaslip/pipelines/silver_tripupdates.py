from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from etaslip.gtfsrt.io import read_snapshot_gz
from etaslip.gtfsrt.parse import decode_feedmessage
from etaslip.gtfsrt.tripupdates import StopTimeRow, iter_stop_time_rows


def silver_output_path(
    silver_root: Path,
    *,
    source_feed: str,
    feed_ts: int,
    part_name: Optional[str] = None,
) -> Path:
    """
    silver/trip_updates/feed=<feed>/dt=YYYY-MM-DD/hour=HH/part-<feed_ts>.parquet
    """
    dt = time.strftime("%Y-%m-%d", time.localtime(feed_ts))
    hh = time.strftime("%H", time.localtime(feed_ts))
    out_dir = (
        silver_root / "trip_updates" / f"feed={source_feed}" / f"dt={dt}" / f"hour={hh}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = part_name or f"part-{feed_ts}"
    return out_dir / f"{suffix}.parquet"


def rows_to_table(rows: Iterable[StopTimeRow]) -> pa.Table:
    """
    Convert StopTimeRow iterable to a PyArrow table with a stable schema.
    """
    rows_list = list(rows)

    schema = pa.schema(
        [
            ("source_feed", pa.string()),
            ("ingest_ts", pa.int64()),
            ("feed_ts", pa.int64()),
            ("trip_id", pa.string()),
            ("route_id", pa.string()),
            ("stop_id", pa.string()),
            ("stop_sequence", pa.int32()),
            ("arrival_time", pa.int64()),
            ("departure_time", pa.int64()),
            ("eta", pa.int64()),
            ("delay_seconds", pa.int32()),
            ("stu_schedule_relationship", pa.string()),
            ("trip_schedule_relationship", pa.string()),
        ]
    )

    def col(name: str):
        return [getattr(r, name) for r in rows_list]

    arrays = [
        pa.array(col("source_feed"), type=pa.string()),
        pa.array(col("ingest_ts"), type=pa.int64()),
        pa.array(col("feed_ts"), type=pa.int64()),
        pa.array(col("trip_id"), type=pa.string()),
        pa.array(col("route_id"), type=pa.string()),
        pa.array(col("stop_id"), type=pa.string()),
        pa.array(col("stop_sequence"), type=pa.int32()),
        pa.array(col("arrival_time"), type=pa.int64()),
        pa.array(col("departure_time"), type=pa.int64()),
        pa.array(col("eta"), type=pa.int64()),
        pa.array(col("delay_seconds"), type=pa.int32()),
        pa.array(col("stu_schedule_relationship"), type=pa.string()),
        pa.array(col("trip_schedule_relationship"), type=pa.string()),
    ]

    return pa.Table.from_arrays(arrays, schema=schema)


def parse_raw_snapshot_to_silver(
    raw_path: Path,
    *,
    silver_root: Path,
    source_feed: str,
    route_filter: Optional[set[str]] = None,
    require_stop_id: bool = True,
    skip_if_exists: bool = True,
) -> Optional[Path]:
    """
    Read one raw .pb.gz snapshot and write one Parquet part to silver.

    Returns output path, or None if skipped/empty.
    """
    pb = read_snapshot_gz(raw_path)
    msg = decode_feedmessage(pb)

    if not msg.header.timestamp:
        return None

    feed_ts = int(msg.header.timestamp)
    out_path = silver_output_path(silver_root, source_feed=source_feed, feed_ts=feed_ts)

    if skip_if_exists and out_path.exists():
        return None

    rows = list(
        iter_stop_time_rows(
            msg,
            source_feed=source_feed,
            route_filter=route_filter,
            require_stop_id=require_stop_id,
        )
    )

    if not rows:
        return None

    table = rows_to_table(rows)
    pq.write_table(table, out_path, compression="zstd")
    return out_path
