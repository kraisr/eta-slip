from __future__ import annotations

from typing import Optional

from google.transit import gtfs_realtime_pb2


def decode_feedmessage(pb: bytes) -> gtfs_realtime_pb2.FeedMessage:
    msg = gtfs_realtime_pb2.FeedMessage()
    msg.ParseFromString(pb)
    return msg


def parse_feed_timestamp(pb: bytes) -> Optional[int]:
    """
    Return FeedHeader.timestamp if present, else None.
    """
    msg = decode_feedmessage(pb)
    ts = getattr(msg.header, "timestamp", None)
    return int(ts) if ts else None
