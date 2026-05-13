"""Imbalance-strategy bake-off + unsupervised baseline.

Trains four candidates on the identical stratified split and logs each
as a separate MLflow run, so the choice of SMOTE is defended with
evidence rather than habit:

  1. xgb_smote            — XGBoost + SMOTE oversampling (production model)
  2. xgb_scale_pos_weight — XGBoost + class weighting (no synthetic data)
  3. xgb_undersample      — XGBoost + random majority undersampling
  4. isolation_forest     — unsupervised anomaly baseline (answers
                            "what if we had no fraud labels?")

Outputs artifacts/model_comparison.json and prints a README-ready
markdown table.

Usage (slow — trains 4 models on the full dataset):
    python -m src.compare_models
"""

import json
import logging
import os
import sys

import numpy as np
from dotenv import load_dotenv
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import mlflow

from src.preprocess import build_dataset, split_features_target
from src.train import RANDOM_STATE, XGB_PARAMS, setup_mlflow

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.compare")

ARTIFACT_DIR = "artifacts"


def classification_metrics(y_test, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
        "average_precision": float(average_precision_score(y_test, y_prob)),
    }


def run_xgb_strategy(name: str, X_train, y_train, X_test, y_test,
                     extra_params: dict | None = None,
                     resampler=None) -> dict:
    """Train one XGBoost variant as its own MLflow run."""
    params = {**XGB_PARAMS, **(extra_params or {})}
    X_fit, y_fit = X_train, y_train
    if resampler is not None:
        X_fit, y_fit = resampler.fit_resample(X_train, y_train)

    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_param("strategy", name)
        mlflow.log_param("train_rows_after_resampling", len(X_fit))

        model = XGBClassifier(**params)
        model.fit(X_fit, y_fit)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = classification_metrics(y_test, y_prob)
        mlflow.log_metrics(metrics)

    logger.info("%-22s AP=%.4f ROC=%.4f P=%.4f R=%.4f F1=%.4f",
                name, metrics["average_precision"], metrics["auc_roc"],
                metrics["precision"], metrics["recall"], metrics["f1"])
    return metrics


def run_isolation_forest(X_train, y_train, X_test, y_test) -> dict:
    """Unsupervised baseline: never sees the labels during fit."""
    with mlflow.start_run(run_name="isolation_forest"):
        params = {
            "n_estimators": 200,
            "contamination": 0.0017,  # known portfolio fraud rate
            "random_state": RANDOM_STATE,
        }
        mlflow.log_params(params)
        mlflow.log_param("strategy", "isolation_forest (unsupervised)")

        forest = IsolationForest(**params, n_jobs=-1)
        forest.fit(X_train)

        # Higher anomaly score = more fraud-like
        anomaly_score = -forest.score_samples(X_test)
        y_pred = (forest.predict(X_test) == -1).astype(int)

        metrics = {
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "auc_roc": float(roc_auc_score(y_test, anomaly_score)),
            "average_precision": float(
                average_precision_score(y_test, anomaly_score)
            ),
        }
        mlflow.log_metrics(metrics)

    logger.info("%-22s AP=%.4f ROC=%.4f P=%.4f R=%.4f F1=%.4f",
                "isolation_forest", metrics["average_precision"],
                metrics["auc_roc"], metrics["precision"], metrics["recall"],
                metrics["f1"])
    return metrics


def to_markdown_table(results: dict) -> str:
    header = ("| Strategy | Avg Precision | AUC-ROC | Precision | Recall | F1 |\n"
              "|---|---|---|---|---|---|")
    rows = [
        f"| {name} | {m['average_precision']:.4f} | {m['auc_roc']:.4f} "
        f"| {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} |"
        for name, m in results.items()
    ]
    return "\n".join([header, *rows])


def compare(csv_path: str) -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    setup_mlflow()

    df = build_dataset(csv_path)
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    results = {
        "xgb_smote": run_xgb_strategy(
            "xgb_smote", X_train, y_train, X_test, y_test,
            resampler=SMOTE(random_state=RANDOM_STATE),
        ),
        "xgb_scale_pos_weight": run_xgb_strategy(
            "xgb_scale_pos_weight", X_train, y_train, X_test, y_test,
            extra_params={"scale_pos_weight": neg / pos},
        ),
        "xgb_undersample": run_xgb_strategy(
            "xgb_undersample", X_train, y_train, X_test, y_test,
            resampler=RandomUnderSampler(random_state=RANDOM_STATE),
        ),
        "isolation_forest": run_isolation_forest(
            X_train, y_train, X_test, y_test
        ),
    }

    best = max(results, key=lambda k: results[k]["average_precision"])
    payload = {
        "best_by_average_precision": best,
        "scale_pos_weight_used": round(neg / pos, 1),
        "results": results,
    }
    with open(os.path.join(ARTIFACT_DIR, "model_comparison.json"), "w") as f:
        json.dump(payload, f, indent=2)

    table = to_markdown_table(results)
    logger.info("Comparison complete — best by average precision: %s", best)
    print("\nREADME-ready table:\n")
    print(table)
    return payload


if __name__ == "__main__":
    try:
        compare(os.getenv("DATA_PATH", "data/creditcard.csv"))
    except Exception:
        logger.exception("Model comparison failed")
        sys.exit(1)
