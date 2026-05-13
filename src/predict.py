"""Inference logic shared by the API and any batch consumers.

Maps a fraud probability to a business risk level and recommended action:

  probability < 0.30  → LOW       → approve
  probability < 0.60  → MEDIUM    → approve, monitor account
  probability < 0.85  → HIGH      → hold for manual review
  otherwise           → CRITICAL  → block and contact cardholder
"""

import logging
import os
from typing import Sequence

import joblib
import numpy as np
from dotenv import load_dotenv

from src.preprocess import MODEL_FEATURES

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.predict")

RISK_BANDS = [
    (0.30, "LOW", "Approve transaction."),
    (0.60, "MEDIUM", "Approve, but flag account for monitoring."),
    (0.85, "HIGH", "Hold transaction for manual review."),
    (1.01, "CRITICAL", "Block transaction and contact cardholder immediately."),
]


def load_model_and_scaler(model_path: str | None = None,
                          scaler_path: str | None = None):
    """Load persisted model + scaler (paths overridable via env)."""
    model_path = model_path or os.getenv("MODEL_PATH", "artifacts/model.pkl")
    scaler_path = scaler_path or os.getenv("SCALER_PATH", "artifacts/scaler.pkl")
    for path in (model_path, scaler_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing — run `python -m src.train` to create artifacts."
            )
    logger.info("Loading model=%s scaler=%s", model_path, scaler_path)
    return joblib.load(model_path), joblib.load(scaler_path)


def risk_from_probability(probability: float) -> tuple[str, str]:
    """Return (risk_level, recommendation) for a fraud probability."""
    for threshold, level, recommendation in RISK_BANDS:
        if probability < threshold:
            return level, recommendation
    return RISK_BANDS[-1][1], RISK_BANDS[-1][2]


def build_feature_vector(amount: float, transaction_hour: int, is_night: int,
                         amount_zscore: float,
                         v1_to_v10: Sequence[float]) -> np.ndarray:
    """Assemble a single row in MODEL_FEATURES order."""
    if len(v1_to_v10) != 10:
        raise ValueError(f"v1_to_v10 must contain 10 values, got {len(v1_to_v10)}")
    row = list(v1_to_v10) + [amount, transaction_hour, is_night, amount_zscore]
    assert len(row) == len(MODEL_FEATURES)
    return np.array([row], dtype=float)


def explain_one(model, scaled_features: np.ndarray, top_k: int = 5) -> list[dict]:
    """SHAP explanation for a single scored transaction.

    Uses XGBoost's native TreeSHAP (`pred_contribs=True`) — exact SHAP
    values for tree models with no extra dependency. Contributions are
    in log-odds space: positive pushes towards fraud, negative away.
    """
    import xgboost as xgb

    booster = model.get_booster()
    contribs = booster.predict(
        xgb.DMatrix(scaled_features), pred_contribs=True
    )[0]
    # Last element is the bias (expected value) — not a feature
    pairs = sorted(
        zip(MODEL_FEATURES, contribs[:-1]),
        key=lambda pair: abs(pair[1]),
        reverse=True,
    )
    return [
        {
            "feature": feature,
            "contribution": round(float(value), 4),
            "direction": "increases_risk" if value > 0 else "decreases_risk",
        }
        for feature, value in pairs[:top_k]
    ]


def predict_one(model, scaler, amount: float, transaction_hour: int,
                is_night: int, amount_zscore: float,
                v1_to_v10: Sequence[float], explain: bool = False) -> dict:
    """Score a single transaction. Returns the API response payload."""
    features = build_feature_vector(
        amount, transaction_hour, is_night, amount_zscore, v1_to_v10
    )
    scaled = scaler.transform(features)
    probability = float(model.predict_proba(scaled)[0, 1])
    risk_level, recommendation = risk_from_probability(probability)

    # Confidence = distance from the maximally-uncertain 0.5 boundary
    confidence = float(abs(probability - 0.5) * 2)

    result = {
        "fraud_probability": round(probability, 6),
        "risk_level": risk_level,
        "confidence": round(confidence, 4),
        "recommendation": recommendation,
    }
    if explain:
        result["top_factors"] = explain_one(model, scaled)
    return result
