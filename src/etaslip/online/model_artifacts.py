from __future__ import annotations

import json
from pathlib import Path


def resolve_model_dir_latest() -> Path:
    latest_ptr = Path("models/eta_slip/latest.txt")
    if latest_ptr.exists():
        p = Path(latest_ptr.read_text().strip())
        if p.exists():
            return p

    root = Path("models/eta_slip")
    if not root.exists():
        raise FileNotFoundError("models/eta_slip not found. Train a model first.")
    runs = sorted(root.glob("run=*"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not runs:
        raise FileNotFoundError("No models found under models/eta_slip/run=*")
    return runs[0]


def load_threshold(model_dir: Path, model_name: str) -> float | None:
    p = model_dir / "metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text())
    try:
        return float(m["models"][model_name]["threshold_selected"])
    except Exception:
        return None
