from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from etaslip.modeling.dataset import DatasetSpec, make_dataset_spec


@dataclass(frozen=True)
class ServingParams:
    arrival_rank: int = 1
    min_lead_sec: int = 480


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


def load_feature_schema(model_dir: Path) -> dict:
    p = model_dir / "feature_schema.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def load_serving_params(model_dir: Path) -> ServingParams:
    schema = load_feature_schema(model_dir)
    serving = schema.get("serving", {}) if isinstance(schema, dict) else {}
    defaults = ServingParams()
    return ServingParams(
        arrival_rank=int(serving.get("arrival_rank", defaults.arrival_rank)),
        min_lead_sec=int(serving.get("min_lead_sec", defaults.min_lead_sec)),
    )


def load_dataset_spec(model_dir: Path) -> DatasetSpec:
    schema = load_feature_schema(model_dir)
    if not schema:
        return DatasetSpec()

    labeling = schema.get("labeling", {}) if isinstance(schema, dict) else {}
    target_mode = str(labeling.get("target_mode", "column"))
    slip_threshold_sec = int(labeling.get("slip_threshold_sec", 120))

    numeric = schema.get("numeric_features")
    categorical = schema.get("categorical_features")
    if not isinstance(numeric, list) or not isinstance(categorical, list):
        return make_dataset_spec(
            target_mode=target_mode,
            slip_threshold_sec=slip_threshold_sec,
            feature_set=str(schema.get("feature_set", "base")),
        )

    return make_dataset_spec(
        target_mode=target_mode,
        slip_threshold_sec=slip_threshold_sec,
        feature_set=str(schema.get("feature_set", "base")),
        numeric_features=tuple(str(c) for c in numeric),
        categorical_features=tuple(str(c) for c in categorical),
    )
