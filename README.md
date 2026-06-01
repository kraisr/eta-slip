# 🚇 ETASlip

Real-time delay risk dashboard for Manhattan southbound **6** train stations.

ETASlip collects MTA GTFS-Realtime TripUpdates, builds training data from ETA
changes, trains a model, and serves a Streamlit dashboard showing which stations
are most likely to see ETA delays soon.

## ✨ What It Does

- Scores the latest GTFS-RT snapshot for covered 6 train stops
- Shows alerts, station risk scores, ETA, and recent alert performance
- Builds `raw -> silver -> gold` data locally
- Trains Logistic Regression and XGBoost models
- Serves the latest model from `models/eta_slip/latest.txt`

## 🧠 Modeling

ETASlip learns from historical ETA changes and live arrival patterns to estimate
delay risk for each station. The training pipeline writes reusable model
artifacts, metrics, thresholds, and feature schema files so the dashboard can
stay aligned with the trained model.

## 📁 Repo Layout

- `app/` - Streamlit dashboard
- `scripts/` - collection, backfill, EDA, and training commands
- `src/etaslip/` - pipeline, modeling, and online feature code
- `config/` - covered stop IDs and stop-name mapping
- `raw/`, `silver/`, `gold/`, `models/` - local generated artifacts

## 🛠️ Setup

```bash
poetry install
```

### Collect Data

```bash
while true; do
  poetry run python scripts/collect_once.py
  sleep 60
done
```

### Build Training Data

```bash
./scripts/backfill.sh
```

### Train

```bash
poetry run python scripts/train_eta_slip.py --gold-path gold/eta_slip --model both
```

### Run Dashboard

```bash
poetry run streamlit run app/dashboard.py
```

Docker:

```bash
docker build -t etaslip:latest .
docker run --rm -p 8501:8501 etaslip:latest
```

> Mount `gold/` and `models/` when running in Docker if you want the dashboard to
show local model artifacts and recent performance charts.
