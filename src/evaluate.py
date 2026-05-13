"""Evaluate the trained model on the held-out test set.

Reads artifacts/model.pkl + artifacts/test_set.npz (written by train.py),
prints a full classification report and saves ROC / Precision-Recall
curve plots into artifacts/ for the dashboard.

Usage:
    python -m src.evaluate
"""

import json
import logging
import os
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics import (
    auc,
    classification_report,
    precision_recall_curve,
    roc_curve,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.evaluate")

ARTIFACT_DIR = "artifacts"


def load_artifacts():
    model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
    test_path = os.path.join(ARTIFACT_DIR, "test_set.npz")
    for path in (model_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing — run `python -m src.train` first.")
    model = joblib.load(model_path)
    data = np.load(test_path)
    return model, data["X_test"], data["y_test"]


def plot_roc(y_test, y_prob, path: str) -> float:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f6feb", label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("FinGuard — ROC Curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return roc_auc


def plot_precision_recall(y_test, y_prob, path: str) -> float:
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall, precision)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#d29922", label=f"PR curve (AUC = {pr_auc:.4f})")
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", alpha=0.4,
               label=f"Baseline (fraud rate = {baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("FinGuard — Precision-Recall Curve")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return pr_auc


def evaluate() -> dict:
    model, X_test, y_test = load_artifacts()
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    logger.info(
        "\n%s",
        classification_report(
            y_test, y_pred, target_names=["Legitimate", "Fraud"], digits=4
        ),
    )

    roc_auc = plot_roc(y_test, y_prob, os.path.join(ARTIFACT_DIR, "roc_curve.png"))
    pr_auc = plot_precision_recall(
        y_test, y_prob, os.path.join(ARTIFACT_DIR, "pr_curve.png")
    )
    logger.info("ROC-AUC=%.4f | PR-AUC=%.4f — plots saved to %s/", roc_auc, pr_auc,
                ARTIFACT_DIR)

    results = {"roc_auc": float(roc_auc), "pr_auc": float(pr_auc)}

    # Merge curve AUCs into the metrics file consumed by /stats
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics.update(results)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return results


if __name__ == "__main__":
    try:
        evaluate()
    except Exception:
        logger.exception("Evaluation failed")
        sys.exit(1)
