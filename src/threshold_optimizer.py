"""Cost-based decision threshold optimization.

The default 0.5 threshold is statistically arbitrary. Fraud decisioning
is a cost trade-off:

  - Missed fraud (false negative)  → lose the full transaction amount
  - False decline (false positive) → fixed customer-friction cost
                                     (industry estimates €15–118; we
                                     default to €30, configurable via
                                     the FP_COST env variable)

This module sweeps thresholds 0.01–0.99 on the held-out test set,
computes total expected cost at each, and picks the minimiser. Outputs:

  artifacts/threshold_analysis.json  — curve + optimal point + savings
  artifacts/cost_curve.png           — cost vs threshold plot
  artifacts/metrics.json             — updated with optimal_threshold

Usage (after training):
    python -m src.threshold_optimizer
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

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.threshold")

ARTIFACT_DIR = "artifacts"
DEFAULT_FP_COST = 30.0  # EUR per falsely declined legitimate transaction


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                     amounts: np.ndarray, fp_cost: float) -> list[dict]:
    """Compute business cost, precision and recall at each threshold.

    Cost(t) = sum(amount of frauds scored below t)   # money lost
            + fp_cost * count(legit scored >= t)     # friction cost
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    amounts = np.asarray(amounts)

    curve = []
    for threshold in np.round(np.arange(0.01, 1.00, 0.01), 2):
        flagged = y_prob >= threshold
        tp = int(np.sum(flagged & (y_true == 1)))
        fp = int(np.sum(flagged & (y_true == 0)))
        fn_mask = (~flagged) & (y_true == 1)
        fn = int(np.sum(fn_mask))

        missed_fraud_cost = float(amounts[fn_mask].sum())
        friction_cost = fp * fp_cost
        total_cost = missed_fraud_cost + friction_cost

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        curve.append({
            "threshold": float(threshold),
            "total_cost": round(total_cost, 2),
            "missed_fraud_cost": round(missed_fraud_cost, 2),
            "friction_cost": round(friction_cost, 2),
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
        })
    return curve


def find_optimal(curve: list[dict]) -> dict:
    return min(curve, key=lambda point: point["total_cost"])


def plot_cost_curve(curve: list[dict], optimal: dict, path: str) -> None:
    thresholds = [p["threshold"] for p in curve]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, [p["total_cost"] for p in curve],
            color="#cf222e", linewidth=2, label="Total business cost")
    ax.plot(thresholds, [p["missed_fraud_cost"] for p in curve],
            color="#d29922", linestyle="--", label="Missed fraud cost")
    ax.plot(thresholds, [p["friction_cost"] for p in curve],
            color="#1f6feb", linestyle="--", label="False-decline friction cost")
    ax.axvline(optimal["threshold"], color="#2da44e", linewidth=2,
               label=f"Optimal t = {optimal['threshold']:.2f}")
    ax.axvline(0.5, color="gray", linestyle=":", label="Default t = 0.50")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Expected cost (EUR, test set)")
    ax.set_title("FinGuard — Cost-Based Threshold Optimization")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def optimize() -> dict:
    fp_cost = float(os.getenv("FP_COST", DEFAULT_FP_COST))

    model_path = os.path.join(ARTIFACT_DIR, "model.pkl")
    test_path = os.path.join(ARTIFACT_DIR, "test_set.npz")
    for path in (model_path, test_path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing — run `python -m src.train` first.")

    model = joblib.load(model_path)
    data = np.load(test_path)
    if "amount_test" not in data:
        raise KeyError(
            "test_set.npz has no amounts — retrain with the current src/train.py."
        )

    y_prob = model.predict_proba(data["X_test"])[:, 1]
    curve = sweep_thresholds(data["y_test"], y_prob, data["amount_test"], fp_cost)
    optimal = find_optimal(curve)
    default = next(p for p in curve if p["threshold"] == 0.50)
    savings = default["total_cost"] - optimal["total_cost"]

    logger.info(
        "Optimal threshold %.2f → cost €%.0f (default 0.50 → €%.0f, saves €%.0f, "
        "%.1f%%) at FP cost €%.0f",
        optimal["threshold"], optimal["total_cost"], default["total_cost"],
        savings, 100 * savings / default["total_cost"] if default["total_cost"] else 0,
        fp_cost,
    )

    plot_cost_curve(curve, optimal,
                    os.path.join(ARTIFACT_DIR, "cost_curve.png"))

    analysis = {
        "fp_cost_assumption_eur": fp_cost,
        "optimal": optimal,
        "default_threshold_point": default,
        "savings_vs_default_eur": round(savings, 2),
        "curve": curve,
    }
    with open(os.path.join(ARTIFACT_DIR, "threshold_analysis.json"), "w") as f:
        json.dump(analysis, f, indent=2)

    # Surface the optimal threshold in metrics.json so /stats can report it
    metrics_path = os.path.join(ARTIFACT_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        metrics["optimal_threshold"] = optimal["threshold"]
        metrics["threshold_savings_eur"] = round(savings, 2)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

    logger.info("Wrote artifacts/threshold_analysis.json and cost_curve.png")
    return analysis


if __name__ == "__main__":
    try:
        optimize()
    except Exception:
        logger.exception("Threshold optimization failed")
        sys.exit(1)
