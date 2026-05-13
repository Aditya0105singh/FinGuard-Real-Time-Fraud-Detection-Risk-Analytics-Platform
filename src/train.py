"""Train the FinGuard fraud detection model.

Pipeline:
  1. Load + preprocess data (src.preprocess)
  2. Stratified 80/20 train/test split
  3. Scale features (StandardScaler) — fit on train only
  4. SMOTE oversampling — applied to training data only
  5. XGBoost classifier
  6. MLflow: log params, metrics, confusion matrix + feature importance
     artifacts, and register the model in the Model Registry
  7. Persist artifacts/model.pkl, artifacts/scaler.pkl, artifacts/metrics.json

Usage:
    python -m src.train [--csv data/creditcard.csv]
"""

import argparse
import json
import logging
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")  # headless environments (Docker, CI)
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import mlflow
import mlflow.sklearn

from src.preprocess import MODEL_FEATURES, build_dataset, split_features_target

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.train")

ARTIFACT_DIR = "artifacts"
MODEL_NAME = "finguard-fraud-xgboost"
RANDOM_STATE = 42

XGB_PARAMS = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


def setup_mlflow() -> None:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "mlruns")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "finguard-fraud-detection")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    logger.info("MLflow tracking → %s (experiment: %s)", tracking_uri, experiment)


def plot_confusion_matrix(y_true, y_pred, path: str) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("FinGuard — Confusion Matrix (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return cm


def plot_feature_importance(model: XGBClassifier, path: str) -> None:
    importance = model.feature_importances_
    order = np.argsort(importance)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(
        [MODEL_FEATURES[i] for i in order],
        importance[order],
        color="#1f6feb",
    )
    ax.set_xlabel("Importance (gain)")
    ax.set_title("FinGuard — Feature Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(csv_path: str) -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    setup_mlflow()

    df = build_dataset(csv_path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    logger.info(
        "Split: train=%d (fraud=%d) | test=%d (fraud=%d)",
        len(X_train), y_train.sum(), len(X_test), y_test.sum(),
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # SMOTE strictly after the split — never let synthetic samples leak
    # into the evaluation set.
    smote = SMOTE(random_state=RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
    logger.info(
        "SMOTE: %d → %d training rows (fraud now %.1f%%)",
        len(X_train_scaled), len(X_resampled), 100 * y_resampled.mean(),
    )

    with mlflow.start_run(run_name="xgboost-smote") as run:
        mlflow.log_params(XGB_PARAMS)
        mlflow.log_param("smote", True)
        mlflow.log_param("test_size", 0.20)
        mlflow.log_param("n_features", len(MODEL_FEATURES))
        mlflow.log_param("features", ",".join(MODEL_FEATURES))

        model = XGBClassifier(**XGB_PARAMS)
        model.fit(X_resampled, y_resampled)

        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, y_prob)),
            "average_precision": float(average_precision_score(y_test, y_prob)),
        }
        mlflow.log_metrics(metrics)
        for name, value in metrics.items():
            logger.info("%-20s %.4f", name, value)

        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
        cm = plot_confusion_matrix(y_test, y_pred, cm_path)
        mlflow.log_artifact(cm_path)

        fi_path = os.path.join(ARTIFACT_DIR, "feature_importance.png")
        plot_feature_importance(model, fi_path)
        mlflow.log_artifact(fi_path)

        mlflow.sklearn.log_model(model, artifact_path="model")
        try:
            mlflow.register_model(
                f"runs:/{run.info.run_id}/model", MODEL_NAME
            )
            logger.info("Model registered as '%s'", MODEL_NAME)
        except Exception as exc:  # registry needs a DB-backed store
            logger.warning("Model Registry unavailable, skipping: %s", exc)

        # Persist serving artifacts
        joblib.dump(model, os.path.join(ARTIFACT_DIR, "model.pkl"))
        joblib.dump(scaler, os.path.join(ARTIFACT_DIR, "scaler.pkl"))
        metrics_payload = {
            **metrics,
            "model_version": "1.0",
            "confusion_matrix": cm.tolist(),
            "features": MODEL_FEATURES,
        }
        with open(os.path.join(ARTIFACT_DIR, "metrics.json"), "w") as f:
            json.dump(metrics_payload, f, indent=2)

        logger.info("Artifacts written to %s/", ARTIFACT_DIR)

    # Hold out the test set for evaluate.py plots and the threshold
    # optimizer (which needs real euro amounts to price missed frauds)
    np.savez_compressed(
        os.path.join(ARTIFACT_DIR, "test_set.npz"),
        X_test=X_test_scaled,
        y_test=y_test.to_numpy(),
        amount_test=X_test["Amount"].to_numpy(),
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the FinGuard fraud model")
    parser.add_argument("--csv", default=os.getenv("DATA_PATH", "data/creditcard.csv"))
    args = parser.parse_args()

    try:
        train(args.csv)
    except Exception:
        logger.exception("Training failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
