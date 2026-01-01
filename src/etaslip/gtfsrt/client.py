from __future__ import annotations

from typing import Optional

import requests


def fetch_feed(feed_url: str, api_key: Optional[str], timeout_sec: int = 20) -> bytes:
    """
    Download a GTFS-RT protobuf feed.

    NOTE: In unit tests, mock requests.get (do not hit the network).
    """
    headers = {
        "User-Agent": "ETASlip/0.1 (collector)",
        "Accept": "application/x-protobuf",
    }
    if api_key:
        headers["x-api-key"] = api_key

    resp = requests.get(feed_url, headers=headers, timeout=timeout_sec)
    resp.raise_for_status()
    return resp.content
