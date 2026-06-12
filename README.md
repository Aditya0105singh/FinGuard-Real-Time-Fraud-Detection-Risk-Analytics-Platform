# 🛡️ FinGuard — Real-Time Fraud Detection & Risk Analytics Platform

An end-to-end ML platform that detects credit card fraud in real time, explains every decision with SHAP, and presents findings through a 6-page analytics dashboard.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1-EB5E28)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-F7931E?logo=scikitlearn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)

[![🚀 Live Dashboard](https://img.shields.io/badge/🚀_Live_Dashboard-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://finguard-real-time-fraud-detection-risk-analytics-platform-muz.streamlit.app/)
[![⚡ FastAPI Docs](https://img.shields.io/badge/⚡_FastAPI_Docs-Swagger_UI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://finguard-real-time-fraud-detection-risk.onrender.com/docs)
[![🏥 API Health](https://img.shields.io/badge/🏥_API_Health-Render-46E3B7?style=for-the-badge&logo=render&logoColor=white)](https://finguard-real-time-fraud-detection-risk.onrender.com/health)
[![📦 GitHub Repo](https://img.shields.io/badge/📦_GitHub-Repository-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/Aditya0105singh/FinGuard-Real-Time-Fraud-Detection-Risk-Analytics-Platform)

> 🔗 **Quick Links:**
> [📊 Live Dashboard](https://finguard-real-time-fraud-detection-risk-analytics-platform-muz.streamlit.app/) &nbsp;|&nbsp;
> [⚡ API Docs](https://finguard-real-time-fraud-detection-risk.onrender.com/docs) &nbsp;|&nbsp;
> [🏥 Health Check](https://finguard-real-time-fraud-detection-risk.onrender.com/health) &nbsp;|&nbsp;
> [📦 GitHub](https://github.com/Aditya0105singh/FinGuard-Real-Time-Fraud-Detection-Risk-Analytics-Platform)

---

## 🚀 Live Demo

| Service | URL |
|---|---|
| 📊 Streamlit Dashboard | [finguard…streamlit.app](https://finguard-real-time-fraud-detection-risk-analytics-platform-muz.streamlit.app/) |
| ⚡ FastAPI Docs (Swagger) | [finguard…onrender.com/docs](https://finguard-real-time-fraud-detection-risk.onrender.com/docs) |
| 🏥 API Health Check | [finguard…onrender.com/health](https://finguard-real-time-fraud-detection-risk.onrender.com/health) |

> **Note:** Render free tier sleeps after 15 min of inactivity — first request may take ~30s to wake up.

---

## 💼 What It Does

Card fraud costs $32B+ annually. Out of 284,807 transactions only 492 (0.17%) are fraudulent — a naive model that approves everything scores 99.8% accuracy while catching zero fraud.

FinGuard treats this as a **business decision problem**:

| Challenge | Solution |
|---|---|
| Extreme class imbalance | SMOTE — choice defended with a 4-way bake-off vs class weighting, undersampling, Isolation Forest |
| 0.5 threshold is arbitrary | Cost-based threshold sweep (missed fraud € vs false-decline friction €) |
| "Model said so" isn't acceptable | Exact TreeSHAP explanation with every prediction |
| Fraud isn't uniform | K-means segmentation isolates high-risk behavioural clusters |

---

## 🏗️ Architecture

```
creditcard.csv ──► preprocess ──► stat tests │ segmentation │ insights
                        │
                        ▼
               train (XGBoost + SMOTE) ──► MLflow runs + registry
                        │
                        ▼
              threshold optimizer ──► artifacts/
                                           │
                        ┌──────────────────┘
                        ▼
               FastAPI /predict (+SHAP) ──► Streamlit Dashboard
```

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| ML | XGBoost, scikit-learn, imbalanced-learn (SMOTE), MLflow |
| Statistics | SciPy (chi-square, Mann-Whitney U, Wilson CIs) |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly |
| Database | PostgreSQL, SQLAlchemy |
| DevOps | Docker Compose, GitHub Actions CI/CD, Render |
| Explainability | TreeSHAP (XGBoost native) |

---

## ⚙️ Run Locally

```bash
git clone https://github.com/Aditya0105singh/FinGuard-Real-Time-Fraud-Detection-Risk-Analytics-Platform.git
cd FinGuard-Real-Time-Fraud-Detection-Risk-Analytics-Platform
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) into `data/`.

```bash
# Run the pipeline
python -m src.train                  # train XGBoost + save artifacts
python -m src.evaluate               # metrics + plots
python -m src.threshold_optimizer    # cost-optimal threshold
python -m src.segmentation           # K-means segments
python -m src.insights               # executive findings

# Serve
uvicorn api.main:app --reload --port 8000   # API → localhost:8000/docs
streamlit run dashboard/app.py              # Dashboard → localhost:8501

# Or with Docker
docker compose up --build
```

---

## 📡 API Reference

**Base URL:** `https://finguard-real-time-fraud-detection-risk.onrender.com`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe |
| `/predict` | POST | Score one transaction + SHAP factors |
| `/batch-predict` | POST | Score up to 1,000 transactions |
| `/stats` | GET | Model metrics from last training run |

**Example:**
```bash
curl -X POST https://finguard-real-time-fraud-detection-risk.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 1.00,
    "transaction_hour": 3,
    "is_night": 1,
    "amount_zscore": -0.35,
    "v1_to_v10": [-3.04, 3.16, -4.30, 4.73, -3.43, -1.64, -5.59, 1.40, -2.54, -4.74]
  }'
```
```json
{
  "fraud_probability": 0.9821,
  "risk_level": "CRITICAL",
  "confidence": 0.9642,
  "recommendation": "Block transaction and contact cardholder immediately.",
  "top_factors": [
    {"feature": "V3", "contribution": -2.14, "direction": "increases_risk"},
    {"feature": "V4", "contribution": 1.87, "direction": "increases_risk"}
  ]
}
```

Risk bands: `< 0.30 LOW` · `< 0.60 MEDIUM` · `< 0.85 HIGH` · `≥ 0.85 CRITICAL`

---

## 📁 Project Structure

```
├── src/
│   ├── train.py                # XGBoost + SMOTE + MLflow
│   ├── predict.py              # inference + risk bands + TreeSHAP
│   ├── preprocess.py           # feature engineering + MODEL_FEATURES contract
│   ├── stat_tests.py           # chi-square, Mann-Whitney U, Wilson CIs
│   ├── threshold_optimizer.py  # cost-based threshold sweep
│   ├── segmentation.py         # K-means fraud profiling
│   ├── compare_models.py       # 4-way imbalance strategy bake-off
│   └── insights.py             # executive findings generator
├── api/                        # FastAPI app + schemas + model loader
├── dashboard/app.py            # 6-page Streamlit dashboard
├── tests/                      # 41 pytest tests
├── sql/queries.sql             # 15 business analytics queries
├── docker-compose.yml
└── requirements.txt
```

---

## ⚖️ License

Dataset: [ULB Machine Learning Group — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (Kaggle).
Built as a portfolio project.
