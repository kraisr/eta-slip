from __future__ import annotations

import os
from dataclasses import dataclass
from zoneinfo import ZoneInfo

NY_TZ = ZoneInfo("America/New_York")

# Not shown in the UI; can be overridden via env var.
DEFAULT_FEED_URL = os.environ.get(
    "GTFSRT_FEED_URL",
    "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds/nyct%2Fgtfs",
)

STOP_IDS_FILE_DEFAULT = "config/stop_ids_manhattan_6_southbound.txt"
STOP_ID_TO_NAME_JSON_DEFAULT = "config/stop_id_to_name.json"


@dataclass(frozen=True)
class Sensitivity:
    label: str
    multiplier: float  # applied to the model's stored threshold


SENSITIVITIES = [
    Sensitivity("High (more alerts)", 0.6),
    Sensitivity("Medium", 1.0),
    Sensitivity("Low (fewer alerts)", 1.5),
]
