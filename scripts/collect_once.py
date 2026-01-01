import argparse
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from etaslip.gtfsrt import fetch_feed, parse_feed_timestamp, write_snapshot

DEFAULT_FEED_URL = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs"
DEFAULT_FEED_NAME = "nyct%2Fgtfs"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--feed-url", default=DEFAULT_FEED_URL)
    p.add_argument("--feed-name", default=DEFAULT_FEED_NAME)
    p.add_argument("--out-root", default="raw/gtfsrt")
    args = p.parse_args()

    load_dotenv()
    api_key = os.getenv("MTA_API_KEY") or None

    pb = fetch_feed(args.feed_url, api_key)
    feed_ts = parse_feed_timestamp(pb)
    out_path = write_snapshot(Path(args.out_root), args.feed_name, pb, feed_ts)

    now = int(time.time())
    age = (now - feed_ts) if feed_ts else None
    print(f"Saved: {out_path}")
    print(f"FeedHeader.timestamp: {feed_ts}")
    print(f"Feed age (sec): {age}")


if __name__ == "__main__":
    main()
