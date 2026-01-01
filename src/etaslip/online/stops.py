from __future__ import annotations

import json
from pathlib import Path

from etaslip.online.constants import STOP_ID_TO_NAME_JSON_DEFAULT


def read_ordered_stop_ids(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Stop IDs file not found: {path}")
    out: list[str] = []
    for line in p.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        out.append(s)
    return out


def load_stop_id_to_name_map() -> dict[str, str]:
    """
    Loads stop_id -> station_name mapping from JSON.
    Tries config/stop_id_to_name.json first; falls back to ./stop_id_to_name.json.
    """
    candidates = [
        Path(STOP_ID_TO_NAME_JSON_DEFAULT),
        Path("stop_id_to_name.json"),
    ]
    for p in candidates:
        if p.exists():
            data = json.loads(p.read_text())
            # ensure str->str
            return {str(k): str(v) for k, v in data.items()}
    return {}
