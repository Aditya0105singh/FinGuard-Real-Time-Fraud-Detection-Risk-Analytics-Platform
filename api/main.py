"""FinGuard — FastAPI fraud scoring API.

When Streamlit Cloud runs this file it automatically redirects to
dashboard/app.py (the actual Streamlit dashboard).

Run the API locally:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys

# Ensure repo root is on sys.path for both api.* and src.* imports.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Streamlit loads streamlit itself into sys.modules before exec-ing this file.
# Uvicorn imports it as a plain module without streamlit present.
# Use that to redirect Streamlit Cloud to the correct entry point.
if "streamlit" in sys.modules:
    _dashboard = os.path.join(_REPO_ROOT, "dashboard", "app.py")
    with open(_dashboard, encoding="utf-8") as _f:
        exec(compile(_f.read(), _dashboard, "exec"),  # noqa: S102
             {"__file__": _dashboard, "__name__": "__main__"})
    sys.exit(0)

# ── FastAPI (uvicorn) path ─────────────────────────────────────────────
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from api.model_loader import MODEL_VERSION, get_bundle
from api.schemas import (
    BatchPredictionInput,
    BatchPredictionOutput,
    HealthResponse,
    PredictionOutput,
    StatsResponse,
    TransactionInput,
)
from src.predict import predict_one

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.api")

app = FastAPI(
    title="FinGuard Fraud Detection API",
    description="Real-time credit card fraud scoring with XGBoost",
    version=MODEL_VERSION,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


def _bundle_or_503():
    """Return the model bundle or fail with a clear 503."""
    try:
        return get_bundle()
    except FileNotFoundError as exc:
        logger.error("Model artifacts missing: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not available. Train the model first "
                   "(python -m src.train).",
        ) from exc


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe used by Docker healthchecks and Render."""
    return HealthResponse(status="healthy", model_version=MODEL_VERSION)


@app.post("/predict", response_model=PredictionOutput, tags=["scoring"])
def predict(transaction: TransactionInput) -> PredictionOutput:
    """Score a single transaction: fraud probability, risk level, and the
    top SHAP factors explaining the decision."""
    bundle = _bundle_or_503()
    try:
        result = predict_one(
            bundle.model,
            bundle.scaler,
            amount=transaction.amount,
            transaction_hour=transaction.transaction_hour,
            is_night=transaction.is_night,
            amount_zscore=transaction.amount_zscore,
            v1_to_v10=transaction.v1_to_v10,
            explain=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal scoring error") from exc

    logger.info(
        "Scored txn amount=%.2f hour=%d → p=%.4f (%s)",
        transaction.amount, transaction.transaction_hour,
        result["fraud_probability"], result["risk_level"],
    )
    return PredictionOutput(**result)


@app.post("/batch-predict", response_model=BatchPredictionOutput, tags=["scoring"])
def batch_predict(batch: BatchPredictionInput) -> BatchPredictionOutput:
    """Score a batch of transactions in a single call."""
    bundle = _bundle_or_503()
    predictions = []
    try:
        for txn in batch.transactions:
            result = predict_one(
                bundle.model,
                bundle.scaler,
                amount=txn.amount,
                transaction_hour=txn.transaction_hour,
                is_night=txn.is_night,
                amount_zscore=txn.amount_zscore,
                v1_to_v10=txn.v1_to_v10,
            )
            predictions.append(PredictionOutput(**result))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail="Internal scoring error") from exc

    logger.info("Batch scored: %d transactions", len(predictions))
    return BatchPredictionOutput(count=len(predictions), predictions=predictions)


@app.get("/stats", response_model=StatsResponse, tags=["ops"])
def stats() -> StatsResponse:
    """Model performance metrics from the most recent training run."""
    bundle = _bundle_or_503()
    if not bundle.metrics:
        raise HTTPException(
            status_code=503,
            detail="Metrics not available — run training to generate metrics.json.",
        )
    m = bundle.metrics
    try:
        return StatsResponse(
            model_version=m.get("model_version", MODEL_VERSION),
            accuracy=m["accuracy"],
            precision=m["precision"],
            recall=m["recall"],
            f1=m["f1"],
            auc_roc=m["auc_roc"],
            average_precision=m["average_precision"],
            optimal_threshold=m.get("optimal_threshold"),
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=500, detail=f"metrics.json missing key: {exc}"
        ) from exc
