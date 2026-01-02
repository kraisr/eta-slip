from __future__ import annotations

import time
from bisect import bisect_left
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

try:
    from zoneinfo import ZoneInfo  # py3.9+
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


NY_TZ = ZoneInfo("America/New_York") if ZoneInfo else None


@dataclass(frozen=True)
class GoldBuildStats:
    snapshots: int
    candidate_examples: int
    matched: int
    missing_at_tplus: int
    no_next_train_at_t: int

    @property
    def match_rate(self) -> float:
        return (
            self.matched / self.candidate_examples if self.candidate_examples else 0.0
        )


def list_silver_files(
    silver_root: Path,
    *,
    source_feed: str,
    dt: str,
    hour: Optional[str] = None,
) -> list[Path]:
    """
    Your silver layout:
      silver/trip_updates/feed=<feed>/dt=YYYY-MM-DD/hour=HH/*.parquet
    """
    base = silver_root / "trip_updates" / f"feed={source_feed}" / f"dt={dt}"
    pattern = f"hour={hour}/*.parquet" if hour is not None else "hour=*/*.parquet"
    return sorted(base.glob(pattern))


def _read_snapshot_table(
    path: Path,
    *,
    route_filter: Optional[set[str]] = None,
    stop_filter: Optional[set[str]] = None,
) -> pa.Table:
    """
    Read one silver Parquet part (one snapshot) and keep only columns needed for labeling.
    """
    cols = ["feed_ts", "route_id", "trip_id", "stop_id", "eta"]
    t = pq.read_table(path, columns=cols)

    # Drop rows with null eta/stop_id
    mask = pc.and_(pc.is_valid(t["eta"]), pc.is_valid(t["stop_id"]))
    t = t.filter(mask)

    if route_filter:
        t = t.filter(pc.is_in(t["route_id"], value_set=pa.array(list(route_filter))))
    if stop_filter:
        t = t.filter(pc.is_in(t["stop_id"], value_set=pa.array(list(stop_filter))))

    return t


def _extract_feed_ts(t: pa.Table) -> int:
    # All rows in a silver part should share the same feed_ts
    arr = t["feed_ts"].to_pylist()
    for v in arr:
        if v is not None:
            return int(v)
    raise ValueError("No feed_ts found in snapshot table.")


def _nearest_time(
    sorted_ts: list[int], target: int
) -> tuple[Optional[int], Optional[int]]:
    """
    Return (nearest_ts, abs_diff). If list is empty -> (None, None)
    """
    if not sorted_ts:
        return None, None

    i = bisect_left(sorted_ts, target)
    candidates = []
    if i < len(sorted_ts):
        candidates.append(sorted_ts[i])
    if i - 1 >= 0:
        candidates.append(sorted_ts[i - 1])

    best = min(candidates, key=lambda x: abs(x - target))
    return best, abs(best - target)


def _dow_hour_minute(feed_ts: int) -> tuple[int, int, int]:
    """
    Local features in NYC time.
    """
    if NY_TZ is None:
        # fallback to localtime if zoneinfo unavailable
        tm = time.localtime(feed_ts)
        return tm.tm_wday, tm.tm_hour, tm.tm_min

    import datetime as _dt

    dt = _dt.datetime.fromtimestamp(feed_ts, tz=NY_TZ)
    return dt.weekday(), dt.hour, dt.minute


def build_gold_eta_slip_for_files(
    silver_files: list[Path],
    *,
    horizon_sec: int = 300,
    tolerance_sec: int = 120,
    slip_threshold_sec: int = 120,
    route_filter: Optional[set[str]] = None,
    stop_filter: Optional[set[str]] = None,
    min_lead_sec: int = 0,
    arrival_rank: int = 1,
) -> tuple[pa.Table, GoldBuildStats]:
    """
    Build gold ETA-slip examples for a set of silver snapshot parts.

    One example per (feed_ts=t, stop_id) where we can find a "next train" at t.
    Labels come from the nearest snapshot to t+horizon_sec within tolerance_sec.
    """
    if not silver_files:
        empty = pa.table(
            {
                "feed_ts": pa.array([], pa.int64()),
                "tplus_feed_ts": pa.array([], pa.int64()),
                "horizon_sec": pa.array([], pa.int32()),
                "stop_id": pa.array([], pa.string()),
                "next_trip_id": pa.array([], pa.string()),
                "eta_t": pa.array([], pa.int64()),
                "eta_t_minutes": pa.array([], pa.float32()),
                "top2_headway_sec": pa.array([], pa.int32()),
                "num_arrivals_listed": pa.array([], pa.int16()),
                "dow": pa.array([], pa.int8()),
                "hour": pa.array([], pa.int8()),
                "minute": pa.array([], pa.int8()),
                "eta_tplus": pa.array([], pa.int64()),
                "slip_seconds": pa.array([], pa.int32()),
                "slip_ge_threshold": pa.array([], pa.int8()),
                "match_status": pa.array([], pa.string()),
            }
        )
        stats = GoldBuildStats(0, 0, 0, 0, 0)
        return empty, stats

    # Read all snapshots into memory for this dt/hour slice
    snapshots: dict[int, pa.Table] = {}
    times: list[int] = []

    for f in silver_files:
        t = _read_snapshot_table(f, route_filter=route_filter, stop_filter=stop_filter)
        if t.num_rows == 0:
            continue
        ts = _extract_feed_ts(t)
        # Deduplicate by feed_ts (keep first)
        if ts not in snapshots:
            snapshots[ts] = t
            times.append(ts)

    times.sort()

    # Output columns as python lists
    out_feed_ts: list[int] = []
    out_tplus_feed_ts: list[Optional[int]] = []
    out_stop_id: list[str] = []
    out_next_trip_id: list[str] = []
    out_eta_t: list[int] = []
    out_eta_t_minutes: list[float] = []
    out_top2_headway_sec: list[Optional[int]] = []
    out_num_arrivals: list[int] = []
    out_dow: list[int] = []
    out_hour: list[int] = []
    out_minute: list[int] = []
    out_eta_tplus: list[Optional[int]] = []
    out_slip_seconds: list[Optional[int]] = []
    out_slip_ge: list[Optional[int]] = []
    out_match_status: list[str] = []

    matched = 0
    missing_at_tplus = 0
    no_next = 0

    for ts in times:
        t = snapshots[ts]
        target = ts + horizon_sec
        tplus_ts, diff = _nearest_time(times, target)

        if tplus_ts is None or diff is None or diff > tolerance_sec:
            # No acceptable label snapshot nearby; skip producing examples for this ts.
            # (You could also emit "stale_feed" rows, but skipping keeps gold clean.)
            continue

        tplus = snapshots[tplus_ts]

        # Build a per-stop list of candidate arrivals at time ts
        # We only consider arrivals with eta >= ts (future relative to snapshot time)
        stop_ids = t["stop_id"].to_pylist()
        trip_ids = t["trip_id"].to_pylist()
        etas = t["eta"].to_pylist()

        per_stop: dict[str, list[tuple[int, str]]] = {}
        for s, trip, eta in zip(stop_ids, trip_ids, etas):
            if s is None or trip is None or eta is None:
                continue
            eta_i = int(eta)
            if eta_i < ts:
                continue
            per_stop.setdefault(str(s), []).append((eta_i, str(trip)))

        # Build lookup at t+Δ: (stop_id, trip_id) -> eta
        tplus_stop = tplus["stop_id"].to_pylist()
        tplus_trip = tplus["trip_id"].to_pylist()
        tplus_eta = tplus["eta"].to_pylist()

        lookup: dict[tuple[str, str], int] = {}
        for s, trip, eta in zip(tplus_stop, tplus_trip, tplus_eta):
            if s is None or trip is None or eta is None:
                continue
            key = (str(s), str(trip))
            # choose min if duplicates
            eta_i = int(eta)
            if key not in lookup or eta_i < lookup[key]:
                lookup[key] = eta_i

        dow, hh, mm = _dow_hour_minute(ts)

        for stop_id, arrivals in per_stop.items():
            arrivals.sort(key=lambda x: x[0])
            if not arrivals:
                no_next += 1
                continue

            eligible = arrivals
            if min_lead_sec > 0:
                cutoff = ts + min_lead_sec
                eligible = [a for a in arrivals if a[0] >= cutoff]

            idx = max(arrival_rank - 1, 0)
            if len(eligible) <= idx:
                no_next += 1
                continue

            eta1, trip1 = eligible[idx]

            # headway feature: still computed from the earliest two arrivals overall
            headway = (arrivals[1][0] - arrivals[0][0]) if len(arrivals) >= 2 else None

            key = (stop_id, trip1)
            eta2 = lookup.get(key)

            out_feed_ts.append(ts)
            out_tplus_feed_ts.append(tplus_ts)
            out_stop_id.append(stop_id)
            out_next_trip_id.append(trip1)
            out_eta_t.append(eta1)
            out_eta_t_minutes.append((eta1 - ts) / 60.0)
            out_top2_headway_sec.append(headway)
            out_num_arrivals.append(len(arrivals))
            out_dow.append(dow)
            out_hour.append(hh)
            out_minute.append(mm)

            if eta2 is None:
                out_eta_tplus.append(None)
                out_slip_seconds.append(None)
                out_slip_ge.append(None)
                out_match_status.append("missing_at_tplus")
                missing_at_tplus += 1
            else:
                slip = int(eta2 - eta1)
                out_eta_tplus.append(int(eta2))
                out_slip_seconds.append(slip)
                out_slip_ge.append(1 if slip >= slip_threshold_sec else 0)
                out_match_status.append("matched")
                matched += 1

    candidate_examples = len(out_feed_ts)

    stats = GoldBuildStats(
        snapshots=len(times),
        candidate_examples=candidate_examples,
        matched=matched,
        missing_at_tplus=missing_at_tplus,
        no_next_train_at_t=no_next,
    )

    table = pa.table(
        {
            "feed_ts": pa.array(out_feed_ts, pa.int64()),
            "tplus_feed_ts": pa.array(out_tplus_feed_ts, pa.int64()),
            "horizon_sec": pa.array([horizon_sec] * candidate_examples, pa.int32()),
            "stop_id": pa.array(out_stop_id, pa.string()),
            "next_trip_id": pa.array(out_next_trip_id, pa.string()),
            "eta_t": pa.array(out_eta_t, pa.int64()),
            "eta_t_minutes": pa.array(out_eta_t_minutes, pa.float32()),
            "top2_headway_sec": pa.array(out_top2_headway_sec, pa.int32()),
            "num_arrivals_listed": pa.array(out_num_arrivals, pa.int16()),
            "dow": pa.array(out_dow, pa.int8()),
            "hour": pa.array(out_hour, pa.int8()),
            "minute": pa.array(out_minute, pa.int8()),
            "eta_tplus": pa.array(out_eta_tplus, pa.int64()),
            "slip_seconds": pa.array(out_slip_seconds, pa.int32()),
            "slip_ge_threshold": pa.array(out_slip_ge, pa.int8()),
            "match_status": pa.array(out_match_status, pa.string()),
        }
    )

    return table, stats


def gold_output_path(
    gold_root: Path,
    *,
    dt: str,
    hour: Optional[str],
    horizon_sec: int,
    source_feed: str,
) -> Path:
    base = (
        gold_root
        / "eta_slip"
        / f"feed={source_feed}"
        / f"horizon_sec={horizon_sec}"
        / f"dt={dt}"
    )
    if hour is not None:
        base = base / f"hour={hour}"
    base.mkdir(parents=True, exist_ok=True)
    part = f"part-{dt}" + (f"-{hour}" if hour is not None else "")
    return base / f"{part}.parquet"


def write_gold_table(table: pa.Table, out_path: Path) -> None:
    pq.write_table(table, out_path, compression="zstd")
