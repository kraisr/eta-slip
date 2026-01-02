from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib

from etaslip.modeling.baselines import (
    baseline_eta_minutes_score,
    baseline_headway_score,
    baseline_no_slip,
)
from etaslip.modeling.dataset import DatasetSpec, load_gold, time_split_3way
from etaslip.modeling.metrics import (
    compute_classification_metrics,
    choose_threshold_max_f1,
    metrics_at_threshold,
)
from etaslip.modeling.trainers import predict_proba, train_logreg, train_xgb

MIN_VAL_POS = 5  # guardrail for tiny validation slices

def _write_latest_pointer(out_root: Path, *, out_root_parent: Path) -> None:
    """
    Atomically write models/eta_slip/latest.txt -> path to the latest run dir.
    We write a *relative* path (from repo root) like: models/eta_slip/run=...
    so it works both locally and in containers (/app/models/...).
    """
    latest_path = out_root_parent / "latest.txt"
    latest_path.parent.mkdir(parents=True, exist_ok=True)

    # Prefer a relative path if possible
    try:
        rel = out_root.relative_to(Path.cwd())
        text = str(rel)
    except Exception:
        text = str(out_root)

    tmp = latest_path.with_suffix(".txt.tmp")
    tmp.write_text(text + "\n")
    tmp.replace(latest_path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-path", required=True, help="Gold parquet file OR directory")
    ap.add_argument("--out-root", default="models/eta_slip")
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--model", choices=["logreg", "xgb", "both"], default="both")

    # Baseline shaping params
    ap.add_argument("--headway-scale-sec", type=float, default=900.0)
    ap.add_argument("--eta-min-min", type=float, default=5.0)
    ap.add_argument("--eta-span-min", type=float, default=20.0)
    ap.add_argument("--min-pred-pos", type=int, default=5, help="Min predicted positives when tuning threshold")

    args = ap.parse_args()

    spec = DatasetSpec()
    df = load_gold(args.gold_path, spec)
    train_df, val_df, test_df = time_split_3way(df, train_frac=args.train_frac, val_frac=args.val_frac)

    if len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0:
        raise SystemExit("Not enough rows after split. Collect more gold data or adjust split fractions.")

    y_train = train_df[spec.target_col].astype(int).to_numpy()
    y_val = val_df[spec.target_col].astype(int).to_numpy()
    y_test = test_df[spec.target_col].astype(int).to_numpy()

    metrics: dict = {
        "data": {
            "gold_path": str(args.gold_path),
            "n_total": int(len(df)),
            "n_train": int(len(train_df)),
            "n_val": int(len(val_df)),
            "n_test": int(len(test_df)),
            "pos_rate_total": float(df[spec.target_col].mean()),
            "pos_rate_train": float(y_train.mean()),
            "pos_rate_val": float(y_val.mean()),
            "pos_rate_test": float(y_test.mean()),
            "train_frac": float(args.train_frac),
            "val_frac": float(args.val_frac),
        },
        "baselines": {},
        "models": {},
    }

    # Baselines (evaluate on TEST)
    metrics["baselines"]["no_slip"] = compute_classification_metrics(y_test, baseline_no_slip(test_df))
    metrics["baselines"]["headway_score"] = compute_classification_metrics(
        y_test, baseline_headway_score(test_df, scale_sec=args.headway_scale_sec)
    )
    metrics["baselines"]["eta_minutes_score"] = compute_classification_metrics(
        y_test, baseline_eta_minutes_score(test_df, min_min=args.eta_min_min, span_min=args.eta_span_min)
    )

    # Output dir
    out_root = Path(args.out_root) / f"run={int(time.time())}"
    out_root.mkdir(parents=True, exist_ok=True)

    # Write feature schema always
    feature_schema = {
        "numeric_features": list(spec.numeric_features),
        "categorical_features": list(spec.categorical_features),
        "target": spec.target_col,
        "match_filter": {spec.match_col: spec.required_match_value},
    }
    (out_root / "feature_schema.json").write_text(json.dumps(feature_schema, indent=2))

    def _fit_and_eval(name: str, pipe) -> None:
        # Decide threshold
        if int(y_val.sum()) >= MIN_VAL_POS:
            prob_val = predict_proba(pipe, val_df, spec)
            thr = choose_threshold_max_f1(y_val, prob_val, min_pred_pos=args.min_pred_pos)
            method = "max_f1_val"
        else:
            if name == "xgb":
                # fallback: tune on TRAIN (not TEST) if val has too few positives
                prob_train = predict_proba(pipe, train_df, spec)
                thr = choose_threshold_max_f1(y_train, prob_train, min_pred_pos=max(args.min_pred_pos, 10))
                method = "max_f1_train_fallback_low_val_pos"
            else:
                thr = 0.5
                method = "fallback_0.5_low_val_pos"

        # Evaluate on TEST
        prob_test = predict_proba(pipe, test_df, spec)
        base = compute_classification_metrics(y_test, prob_test)
        tuned = metrics_at_threshold(y_test, prob_test, thr)

        base["threshold_selected"] = float(thr)
        base["threshold_selection_method"] = method
        base["tuned_at_threshold_selected"] = tuned

        metrics["models"][name] = base
        joblib.dump(pipe, out_root / f"model_{name}.joblib")

    # Train on TRAIN only
    if args.model in ("logreg", "both"):
        pipe = train_logreg(train_df, spec)
        _fit_and_eval("logreg", pipe)

    if args.model in ("xgb", "both"):
        pipe = train_xgb(train_df, spec)
        _fit_and_eval("xgb", pipe)

    (out_root / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_latest_pointer(out_root, out_root_parent=Path(args.out_root))
    print(f"Saved artifacts to: {out_root}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
