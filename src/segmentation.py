"""Transaction segmentation — K-means clustering with fraud profiling.

Clusters transactions on behavioural features (log amount, hour of day,
top PCA components), then profiles each segment: size, fraud rate,
share of all fraud, and risk multiple vs the portfolio baseline.

The headline output is the kind of finding analytics teams are hired
to produce: "segment X is N% of volume but M% of fraud."

Outputs:
  artifacts/segments.json  — per-segment profile + silhouette score
  artifacts/segments.png   — cluster scatter (hour × log-amount)

Usage:
    python -m src.segmentation [--clusters 4]
"""

import argparse
import json
import logging
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.preprocess import build_dataset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.segmentation")

ARTIFACT_DIR = "artifacts"
CLUSTER_FEATURES = ["amount_log", "transaction_hour"] + [f"V{i}" for i in range(1, 11)]
RANDOM_STATE = 42
SILHOUETTE_SAMPLE = 10_000  # silhouette is O(n²); score on a sample


def profile_segments(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    df = df.assign(segment=labels)
    baseline_rate = df["Class"].mean()
    total_fraud = df["Class"].sum()

    profiles = []
    for segment, group in df.groupby("segment"):
        fraud_count = int(group["Class"].sum())
        fraud_rate = group["Class"].mean()
        profiles.append({
            "segment": int(segment),
            "size": len(group),
            "pct_of_transactions": round(100 * len(group) / len(df), 2),
            "fraud_count": fraud_count,
            "fraud_rate_pct": round(100 * fraud_rate, 4),
            "pct_of_all_fraud": round(100 * fraud_count / total_fraud, 2)
                                if total_fraud else 0.0,
            "risk_multiple_vs_baseline": round(fraud_rate / baseline_rate, 2)
                                         if baseline_rate else 0.0,
            "avg_amount": round(float(group["Amount"].mean()), 2),
            "median_amount": round(float(group["Amount"].median()), 2),
            "night_share_pct": round(100 * group["is_night"].mean(), 2),
            "fraud_amount_eur": round(
                float(group.loc[group["Class"] == 1, "Amount"].sum()), 2
            ),
        })
    return sorted(profiles, key=lambda p: p["risk_multiple_vs_baseline"],
                  reverse=True)


def plot_segments(df: pd.DataFrame, labels: np.ndarray, path: str) -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(df), size=min(30_000, len(df)), replace=False)
    sample = df.iloc[sample_idx]
    sample_labels = labels[sample_idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    # Jitter hours so the integer column reads as a density
    jitter = rng.uniform(-0.4, 0.4, size=len(sample))
    scatter = ax.scatter(
        sample["transaction_hour"] + jitter, sample["amount_log"],
        c=sample_labels, cmap="tab10", s=4, alpha=0.4,
    )
    frauds = sample[sample["Class"] == 1]
    fraud_jitter = rng.uniform(-0.4, 0.4, size=len(frauds))
    ax.scatter(frauds["transaction_hour"] + fraud_jitter, frauds["amount_log"],
               marker="x", color="red", s=40, label="Fraud")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("log1p(Amount)")
    ax.set_title("FinGuard — Transaction Segments (K-means)")
    ax.legend(loc="upper right")
    fig.colorbar(scatter, label="Segment")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def segment(csv_path: str, n_clusters: int = 4) -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = build_dataset(csv_path)

    X = StandardScaler().fit_transform(df[CLUSTER_FEATURES])
    logger.info("Clustering %d transactions into %d segments ...", len(X), n_clusters)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=RANDOM_STATE)
    labels = kmeans.fit_predict(X)

    rng = np.random.default_rng(RANDOM_STATE)
    sil_idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
    silhouette = float(silhouette_score(X[sil_idx], labels[sil_idx]))
    logger.info("Silhouette score (%d-row sample): %.3f", len(sil_idx), silhouette)

    profiles = profile_segments(df, labels)
    for p in profiles:
        logger.info(
            "Segment %d: %5.1f%% of txns | fraud rate %.4f%% | %5.1f%% of all "
            "fraud | %.1fx baseline risk",
            p["segment"], p["pct_of_transactions"], p["fraud_rate_pct"],
            p["pct_of_all_fraud"], p["risk_multiple_vs_baseline"],
        )

    plot_segments(df, labels, os.path.join(ARTIFACT_DIR, "segments.png"))

    payload = {
        "n_clusters": n_clusters,
        "features_used": CLUSTER_FEATURES,
        "silhouette_score": round(silhouette, 4),
        "baseline_fraud_rate_pct": round(100 * float(df["Class"].mean()), 4),
        "segments": profiles,
    }
    with open(os.path.join(ARTIFACT_DIR, "segments.json"), "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote artifacts/segments.json and segments.png")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster transactions into segments")
    parser.add_argument("--csv", default=os.getenv("DATA_PATH", "data/creditcard.csv"))
    parser.add_argument("--clusters", type=int, default=4)
    args = parser.parse_args()
    try:
        segment(args.csv, args.clusters)
    except Exception:
        logger.exception("Segmentation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
