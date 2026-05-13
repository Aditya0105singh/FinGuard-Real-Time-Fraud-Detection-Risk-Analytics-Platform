"""Shared fixtures: train a tiny model into a temp artifacts dir so the
API and model tests run without the real dataset or prior training."""

import json
import os
import sys

import joblib
import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Make `src` and `api` importable when pytest runs from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import MODEL_FEATURES  # noqa: E402

N_FEATURES = len(MODEL_FEATURES)


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory):
    """Train a minimal XGBoost model on synthetic data and persist the
    same artifact set that src.train produces."""
    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(size=(n, N_FEATURES))
    # Synthetic signal: fraud correlates with the first feature
    y = (X[:, 0] + rng.normal(scale=0.5, size=n) > 1.2).astype(int)

    scaler = StandardScaler().fit(X)
    model = XGBClassifier(
        n_estimators=10, max_depth=3, learning_rate=0.3, random_state=42
    )
    model.fit(scaler.transform(X), y)

    out = tmp_path_factory.mktemp("artifacts")
    joblib.dump(model, out / "model.pkl")
    joblib.dump(scaler, out / "scaler.pkl")
    metrics = {
        "model_version": "1.0",
        "accuracy": 0.99, "precision": 0.9, "recall": 0.85,
        "f1": 0.87, "auc_roc": 0.97, "average_precision": 0.8,
        "confusion_matrix": [[450, 5], [10, 35]],
        "features": MODEL_FEATURES,
    }
    (out / "metrics.json").write_text(json.dumps(metrics))
    return out


@pytest.fixture(scope="session")
def api_client(artifacts_dir):
    """TestClient wired to the temp artifacts via env vars."""
    os.environ["MODEL_PATH"] = str(artifacts_dir / "model.pkl")
    os.environ["SCALER_PATH"] = str(artifacts_dir / "scaler.pkl")
    os.environ["METRICS_PATH"] = str(artifacts_dir / "metrics.json")

    from fastapi.testclient import TestClient

    from api import model_loader
    from api.main import app

    model_loader.get_bundle.cache_clear()
    with TestClient(app) as client:
        yield client


@pytest.fixture()
def valid_payload():
    return {
        "amount": 149.62,
        "transaction_hour": 2,
        "is_night": 1,
        "amount_zscore": 0.24,
        "v1_to_v10": [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09],
    }
