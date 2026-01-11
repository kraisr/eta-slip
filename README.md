<h1 align="center">🚇 ETASlip — Real-time 6 Train Delay Risk (NYC)</h1>

ETASlip is an end-to-end ML system that continuously collects MTA TripUpdates using GTFS-Realtime API, builds a labeled dataset (“ETA slip” events), trains a model, and serves a live dashboard for Manhattan southbound **6** train stations.

The dashboard helps you determine:
**“Which stations are most likely to experience an ETA slip soon?”**

---

## What it does

### 🔴 Live “Delay Risk” dashboard
- Fetches the latest GTFS-Realtime TripUpdates and scores the current snapshot
- Shows a clean view:
  - **Alerts** (stations above an alert threshold)
  - A **line strip** view of stations (ordered by travel direction)
  - **Search-as-you-type** station search
- Converts stop IDs into readable **station names** using `config/stop_id_to_name.json`

### 🧱 Continuous data pipeline (raw → silver → gold)
- **Raw**: stores the original GTFS-RT snapshots (`.pb`/`.pb.gz`) organized by date/hour
- **Silver**: parses TripUpdates into structured parquet
- **Gold**: builds labeled examples for “ETA slip within horizon” for the 6 train Manhattan core stops  
  (includes match rates, missing-at-t+ stats, and slip rate diagnostics)

### 🤖 Training and model artifacts
- Trains baseline models and ML models (Logistic Regression, XGBoost)
- Produces reproducible run artifacts:
  - `metrics.json` with PR AUC / ROC AUC / threshold selection details
  - serialized models (`model_xgb.joblib`, `model_logreg.joblib`)
  - `feature_schema.json` describing feature columns and filters
- Maintains a stable pointer for the serving layer:
  - `models/eta_slip/latest.txt` points to the most recent trained run folder

### ☁️ Deployment + automation (AWS EC2 + S3 + GitHub Actions)
- Runs the app and long-running collectors in docker containers on **EC2**
- Uses **S3** as durable storage for raw data and training artifacts
- Automates new data generation, retraining and redeploy via **GitHub Actions** and **AWS SSM**
- Scheduled retraining workflow (example: every 4 days at 4AM ET)

<details>
  <summary>Example AWS production loop:</summary>

  - Collector container write raw snapshots to disk and syncs to S3 periodically
  - Retrain script:
    - sync raw data from S3
    - rebuild silver + gold data
    - train new model
    - sync model back to S3
    - restart dashboard container

(Deployment and retraining can be triggered on EC2 via AWS SSM from GitHub Actions.)
</details>


---

## 📁 Repository layout

- `scripts/`
  - data collection, parsing to silver, gold labeling, training, EDA utilities
- `src/etaslip/`
  - dataset spec, preprocessing, trainers, metrics, baselines
- `app/`
  - Streamlit dashboard and UI helpers
- `config/`
  - stop list for covered stations, stop ID → station name mapping
- `raw/`, `silver/`, `gold/`, `models/`
  - local artifacts (typically not committed; synced to S3 in production)

---

## 📊 Data + labeling (ETA slip)
ETASlip generates supervised labels from real GTFS-RT snapshots by:
- selecting arrivals with a minimum lead time (to avoid unstable near-arrival updates)
- tracking how the ETA changes over time for the same trip/stop
- labeling whether an ETA “slips” beyond a threshold within a prediction horizon

This produces a dataset where each row corresponds to a station snapshot with:
- current ETA (minutes)
- headway signals (seconds)
- count of upcoming arrivals observed
- time context (day of week / hour / minute)
- binary target: slip/no-slip within the configured horizon

---

## Model serving behavior
- The app loads the **latest** model run (via `models/eta_slip/latest.txt` or newest `run=*`)
- Live scoring:
  - fetch → parse → filter to covered 6-train southbound stops → feature build → predict probabilities
- “Alert sensitivity” adjusts the effective threshold used to flag stations

---

# 🛠️ Instructions

## 1) Local setup
Prereqs:
- Python 3.12+
- Poetry 2.x

```bash
poetry install
```

## 2) Collect raw data
Run an infinite loop that collects a snapshot every 60s:

```bash
while true; do
  poetry run python scripts/collect_once.py
  sleep 60
done
```

## 3) Generate silver + gold from collected raw data

```bash
bash scripts/backfill.sh
```

## 4) Train a model
Train on all discovered gold data:

```bash
poetry run python scripts/train_eta_slip.py \
  --gold-path gold/eta_slip/feed=nyct%2Fgtfs/horizon_sec=300 \
  --train-frac 0.6 \
  --val-frac 0.2 \
  --min-pred-pos 5 \
  --model both
```

## 5) Run the dashboard
Locally:

```bash
poetry run streamlit run app/dashboard.py
```

or using docker:

```bash
docker build -t etaslip:latest .
docker run --rm -p 8501:8501 etaslip:latest
```

> Note: If docker is used to run the dashboard, the gold data needs to be mounted for the precision graph to show

