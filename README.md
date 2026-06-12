# 🛡️ FinGuard — Real-Time Fraud Detection & Risk Analytics Platform

> An end-to-end fintech ML platform that ingests 284K+ card transactions into
> PostgreSQL, validates fraud patterns with formal statistical tests, trains an
> XGBoost classifier (defended against three alternative imbalance strategies),
> tunes its decision threshold on a real cost model, and serves explainable
> SHAP-backed risk scores through a FastAPI microservice with a 6-page
> Streamlit analytics dashboard — fully containerised with CI/CD to Render.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-4169E1?logo=postgresql&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-EB5E28)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.14-8CAAE6?logo=scipy&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.17-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

---

## 💼 Business Problem

Card fraud costs the global payments industry **$32+ billion annually**, and
every false decline costs merchants real money in lost lifetime value. The
core difficulty is brutal asymmetry: in this dataset only **492 of 284,807
transactions (0.172%)** are fraudulent, so a naive model that approves
everything is 99.8% "accurate" while catching zero fraud.

FinGuard treats fraud detection as a **business decision problem**, not just
a classification problem:

| Challenge | FinGuard's answer |
|---|---|
| Extreme class imbalance | SMOTE on training data only — choice defended with a 4-way bake-off |
| Accuracy is meaningless | Evaluated on PR-AUC, recall, precision; statistical tests behind every claim |
| 0.5 threshold is arbitrary | Cost-based threshold sweep (missed fraud € vs false-decline friction €) |
| "The model said so" isn't acceptable | Exact TreeSHAP explanation returned with every prediction |
| Fraud isn't uniform | K-means segmentation isolates a small high-risk behavioural segment |
| Charts aren't decisions | Executive Insights page: quantified findings + recommendations |

## 🏗️ Architecture

```
                        ┌──────────────────┐
                        │  creditcard.csv  │  Kaggle dataset (284,807 txns)
                        └────────┬─────────┘
                                 │  src/ingest.py (chunked load + enrichment)
                                 ▼
 ┌──────────────┐       ┌──────────────────┐       ┌──────────────────┐
 │ sql/         │◄──────│   PostgreSQL     │       │  src/preprocess  │
 │ 15 analytics │       │  `transactions`  │──────►│  feature         │
 │ queries      │       │  (indexed)       │       │  engineering     │
 └──────────────┘       └──────────────────┘       └────────┬─────────┘
                                                            │
        ┌──────────────────┬────────────────────────────────┤
        ▼                  ▼                                ▼
 ┌─────────────┐   ┌──────────────┐   ┌────────────────────────────────┐
 │ stat_tests  │   │ segmentation │   │ train.py  (SMOTE + XGBoost)    │
 │ chi², MWU,  │   │ K-means +    │   │ compare_models.py (4-way       │
 │ Wilson CIs  │   │ fraud profile│   │   bake-off, MLflow runs)       │
 └──────┬──────┘   └──────┬───────┘   │ threshold_optimizer.py (cost)  │
        │                 │           └───────┬──────────────┬─────────┘
        │                 │                   │              │
        │                 │        ┌──────────▼────┐  ┌──────▼──────────┐
        │                 │        │    MLflow     │  │   artifacts/    │
        │                 │        │ runs, registry│  │ model, scaler,  │
        │                 │        └───────────────┘  │ metrics, curves │
        │                 │                           └──────┬──────────┘
        └────────┬────────┘                                  │
                 ▼                                           │
        ┌────────────────┐                                   │
        │  insights.py   │  quantified findings + recs       │
        └────────┬───────┘                                   │
                 │              ┌────────────────────────────▼────────┐
                 │              │        FastAPI  (port 8000)         │
                 │              │ /health /predict /batch-predict     │
                 │              │ /stats  — predictions include SHAP  │
                 │              └────────────────┬────────────────────┘
                 │                               │ REST
        ┌────────▼───────────────────────────────▼────────────────────┐
        │              Streamlit Dashboard (port 8501)                │
        │ Overview │ Fraud Analysis │ Segments │ Executive Insights   │
        │          │ Model Performance │ Live Prediction (SHAP)       │
        └─────────────────────────────────────────────────────────────┘

        Docker Compose: postgres + mlflow + api + dashboard
        GitHub Actions: pytest (41 tests) → docker build → Render deploy
```

## ⚙️ How It Works — Component Walkthrough

### 1. Ingestion ([src/ingest.py](src/ingest.py))
Streams the 150 MB CSV into PostgreSQL in 50K-row chunks (constant memory),
adding business columns during the load: `fraud_label`, amount-based
`risk_tier`, `transaction_hour`, `day_of_week`, and a synthetic
`transaction_timestamp` (the dataset only records seconds-elapsed, so date
queries need an anchor). Creates indexes on amount, class, time.

### 2. SQL analytics ([sql/queries.sql](sql/queries.sql))
15 production-style queries: fraud rates by hour/day, quartile segmentation
(`NTILE`), running counts (`ROW_NUMBER`), rolling averages (`LAG`),
cumulative loss curves (`SUM OVER`), within-group ranking (`RANK`),
peak-hour detection (`HAVING`), and YTD growth reporting (CTE + `LAG`).

### 3. Statistical validation ([src/stat_tests.py](src/stat_tests.py))
Every EDA claim is backed by a formal test:
- **Chi-square test of independence** — confirms night-time fraud elevation
  is significant, not noise (justifies the `is_night` feature and a step-up
  authentication policy).
- **Mann-Whitney U** — fraud vs legitimate amount distributions differ.
  Chosen over a t-test because Amount is extremely right-skewed (normality
  fails); MWU is rank-based and distribution-free.
- **Wilson score intervals** — per-hour fraud rates with proper uncertainty.
  At a 0.17% base rate the normal-approximation CI can go negative; Wilson
  behaves correctly near zero.

### 4. Feature engineering ([src/preprocess.py](src/preprocess.py))
Engineered features: `transaction_hour`, `amount_log` (log1p — heavy right
skew), `amount_zscore`, `is_night`, `is_high_amount` (75th percentile flag),
`time_diff`. The module exports **`MODEL_FEATURES`** — the exact feature
contract (V1–V10 + Amount + hour + is_night + zscore) imported by training,
inference, and the API, so train/serve skew is structurally impossible.

### 5. Training ([src/train.py](src/train.py))
Stratified 80/20 split → StandardScaler (fit on train only) → **SMOTE on the
training split only** (synthetic samples must never leak into evaluation) →
XGBoost (100 trees, depth 6, lr 0.1). MLflow logs every hyperparameter, six
metrics, confusion-matrix and feature-importance artifacts, and registers
the model. Persists `model.pkl`, `scaler.pkl`, `metrics.json`, and the
held-out test set for downstream analysis.

### 6. Imbalance bake-off ([src/compare_models.py](src/compare_models.py))
SMOTE is a choice, not a default. Four candidates trained on the identical
split as separate MLflow runs: **SMOTE**, **scale_pos_weight** (class
weighting), **random undersampling**, and an **unsupervised Isolation
Forest** (answers "what if we had no fraud labels?"). Results:

| Strategy | Avg Precision | AUC-ROC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| xgb_smote | 0.7460 | 0.9671 | 0.2459 | 0.7895 | 0.3750 |
| **xgb_scale_pos_weight** | **0.7589** | 0.9648 | **0.5252** | 0.7684 | **0.6239** |
| xgb_undersample | 0.6025 | 0.9692 | 0.0478 | 0.8737 | 0.0907 |
| isolation_forest | 0.0509 | 0.8937 | 0.1200 | 0.1263 | 0.1231 |

Notable result: **class weighting beat SMOTE** on average precision and more
than doubled precision at equal recall — evidence that synthetic
oversampling added boundary noise here. The unsupervised Isolation Forest's
0.05 AP vs 0.76 supervised quantifies the value of having labels.

### 7. Cost-based threshold ([src/threshold_optimizer.py](src/threshold_optimizer.py))
Sweeps thresholds 0.01–0.99 against a cost model: **missed fraud costs the
full transaction amount; a false decline costs €30 friction** (configurable
via `FP_COST`; industry estimates range €15–118). Picks the cost-minimising
operating point, quantifies savings vs the 0.5 default, and writes the cost
curve to the dashboard.

### 8. Segmentation ([src/segmentation.py](src/segmentation.py))
K-means (k=4, silhouette-scored) on log-amount, hour, and V1–V10. Profiles
each segment — size, fraud rate, share of all fraud, risk multiple vs
baseline, fraud € value — surfacing findings like *"segment X is N% of
volume but M% of fraud."*

### 9. Executive insights ([src/insights.py](src/insights.py))
Aggregates the tests, threshold analysis, segmentation, and model metrics
into 5–7 quantified findings, each with a concrete recommendation (e.g.
*"apply step-up authentication to night transactions"*). Rendered as a memo
on the dashboard — analysis a fraud-operations manager could act on.

### 10. Explainable serving ([api/](api/), [src/predict.py](src/predict.py))
`/predict` returns probability, LOW/MEDIUM/HIGH/CRITICAL risk band, a
recommendation, **and the top-5 SHAP factors** that drove the score.
SHAP values are computed with XGBoost's native exact TreeSHAP
(`pred_contribs=True`) — no extra dependency, sub-millisecond overhead.
Pydantic validates every input (exactly 10 PCA values, hour 0–23,
non-negative amount); errors map to clean HTTP 422/503/500 responses.

### 11. Dashboard ([dashboard/app.py](dashboard/app.py))
Six pages: **Overview** (KPIs, volume, fraud ratio), **Fraud Analysis**
(hourly rates, hour×day heatmap, amount distributions), **Segments**
(cluster profiles + risk chart), **Executive Insights** (findings memo),
**Model Performance** (metrics, cost curve, bake-off table, confusion
matrix, ROC/PR), **Live Prediction** (form → API → risk gauge + SHAP
waterfall). Reads PostgreSQL when available, falls back to CSV.

## 🚀 Live Demo

| Service | URL |
|---|---|
| 📊 Streamlit Dashboard | _Coming soon — deploying to Render_ |
| ⚡ FastAPI Docs (Swagger) | _Coming soon — deploying to Render_ |

## ⚙️ Setup & Run

### 1. Install

```bash
git clone https://github.com/Aditya0105singh/FinGuard-Real-Time-Fraud-Detection-Risk-Analytics-Platform.git
cd FinGuard-Real-Time-Fraud-Detection-Risk-Analytics-Platform
python -m venv .venv
# Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then edit credentials
```

### 2. Get the dataset
Download `creditcard.csv` from Kaggle into `data/`:
[Credit Card Fraud Detection — Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### 3. Run the analytics pipeline (in order)

```bash
python -m src.ingest                 # CSV → PostgreSQL (optional, needs DB)
python -m src.train                  # SMOTE + XGBoost + MLflow
python -m src.evaluate               # report + ROC/PR plots
python -m src.threshold_optimizer    # cost-optimal threshold + curve
python -m src.segmentation           # K-means segments + profiles
python -m src.insights               # executive findings memo
python -m src.compare_models         # 4-way bake-off (slow, optional)
```

### 4. Serve locally

```bash
uvicorn api.main:app --reload --port 8000     # API → localhost:8000/docs
streamlit run dashboard/app.py                # Dashboard → localhost:8501
pytest tests/ -v                              # 41 tests
```

### Or everything at once with Docker

```bash
docker compose up --build
# PostgreSQL :5432 | MLflow :5000 | API :8000/docs | Dashboard :8501
```

## 📡 API Reference

Interactive docs at `http://localhost:8000/docs`.

### `POST /predict`
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 149.62,
    "transaction_hour": 2,
    "is_night": 1,
    "amount_zscore": 0.24,
    "v1_to_v10": [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09]
  }'
```
```json
{
  "fraud_probability": 0.0312,
  "risk_level": "LOW",
  "confidence": 0.9376,
  "recommendation": "Approve transaction.",
  "top_factors": [
    {"feature": "V4", "contribution": 1.21, "direction": "increases_risk"},
    {"feature": "V1", "contribution": -0.87, "direction": "decreases_risk"},
    {"feature": "is_night", "contribution": 0.43, "direction": "increases_risk"}
  ]
}
```

Risk bands: `< 0.30 LOW` · `< 0.60 MEDIUM` · `< 0.85 HIGH` · `≥ 0.85 CRITICAL`

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness probe (Docker healthcheck / Render) |
| `POST /predict` | Single transaction → score + risk + SHAP factors |
| `POST /batch-predict` | Up to 1,000 transactions (explanations omitted for throughput) |
| `GET /stats` | Last training run's metrics + cost-optimal threshold |

## 🔁 CI/CD

Every push to `main`: **test** (41 pytest tests on synthetic fixtures — no
dataset needed in CI) → **build** (both Docker images, layer-cached) →
**deploy** (Render deploy hook via `RENDER_DEPLOY_HOOK_URL` secret).

## 📁 Project Structure

```
fintech-fraud-detection/
├── .github/workflows/ci-cd.yml   # test → build → deploy
├── sql/queries.sql               # 15 business analytics queries
├── src/
│   ├── ingest.py                 # CSV → PostgreSQL + enrichment + indexes
│   ├── preprocess.py             # cleaning + MODEL_FEATURES contract
│   ├── stat_tests.py             # chi-square, Mann-Whitney U, Wilson CIs
│   ├── train.py                  # SMOTE + XGBoost + MLflow
│   ├── evaluate.py               # report + ROC/PR plots
│   ├── threshold_optimizer.py    # cost-based threshold sweep
│   ├── segmentation.py           # K-means + fraud profiling
│   ├── insights.py               # executive findings generator
│   ├── compare_models.py         # 4-way imbalance bake-off
│   └── predict.py                # inference + risk bands + TreeSHAP
├── api/                          # FastAPI app, schemas, model loader
├── dashboard/app.py              # 6-page Streamlit dashboard
├── notebooks/eda.ipynb           # EDA incl. hypothesis testing
├── tests/                        # 41 tests (API, model, analytics)
├── docker-compose.yml            # postgres + mlflow + api + dashboard
├── Dockerfile.api / Dockerfile.dashboard
└── requirements.txt              # pinned dependencies
```

## ⚖️ License & Attribution

Dataset: [ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle).
Built as a portfolio project; not affiliated with any financial institution.
