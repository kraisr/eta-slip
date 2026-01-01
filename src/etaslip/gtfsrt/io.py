from __future__ import annotations

import gzip
import time
from pathlib import Path
from typing import Optional


def snapshot_output_path(out_root: Path, feed_name: str, ts_epoch: int) -> Path:
    """
    Partition by dt/hour only (no min partition).
    """
    dt = time.strftime("%Y-%m-%d", time.localtime(ts_epoch))
    hh = time.strftime("%H", time.localtime(ts_epoch))
    out_dir = out_root / f"feed={feed_name}" / f"dt={dt}" / f"hour={hh}"
    return out_dir / f"{ts_epoch}.pb.gz"


def write_snapshot(out_root: Path, feed_name: str, pb: bytes, feed_ts: Optional[int]) -> Path:
    """
    Write a gzipped protobuf snapshot.

    Naming: prefer FeedHeader.timestamp; fallback to fetch-time.
    Layout: raw/gtfsrt/feed=<feed>/dt=YYYY-MM-DD/hour=HH/<ts>.pb.gz
    """
    now = int(time.time())
    ts = int(feed_ts) if feed_ts else now

    out_path = snapshot_output_path(out_root=out_root, feed_name=feed_name, ts_epoch=ts)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic-ish write: write temp then rename
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    with gzip.open(tmp_path, "wb") as f:
        f.write(pb)
    tmp_path.replace(out_path)

    return out_path


def read_snapshot_gz(path: Path) -> bytes:
    """
    Read a gzipped protobuf snapshot into bytes.
    """
    with gzip.open(path, "rb") as f:
        return f.read()
