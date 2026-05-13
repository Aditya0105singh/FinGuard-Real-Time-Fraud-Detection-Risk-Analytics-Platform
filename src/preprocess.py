"""Cleaning and feature engineering for the fraud detection model.

Engineered features:
  - transaction_hour : hour of day extracted from the Time offset
  - amount_log       : log1p transform of Amount (heavy right skew)
  - amount_zscore    : z-score normalised Amount
  - is_night         : 1 if hour in [0, 6]
  - is_high_amount   : 1 if amount above the 75th percentile
  - time_diff        : seconds since the previous transaction

MODEL_FEATURES defines the exact column order the trained model expects;
train.py, predict.py and the API all import it from here.
"""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.preprocess")

# Feature contract shared by training and serving. The API exposes the
# top-10 PCA components plus the engineered amount/time features.
MODEL_FEATURES = [f"V{i}" for i in range(1, 11)] + [
    "Amount",
    "transaction_hour",
    "is_night",
    "amount_zscore",
]
TARGET = "Class"

NIGHT_START, NIGHT_END = 0, 6
HIGH_AMOUNT_QUANTILE = 0.75


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates and rows with missing critical values."""
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna(subset=["Time", "Amount", TARGET])
    logger.info("Cleaning removed %d rows (%d remain)", before - len(df), len(df))
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all engineered features. Expects raw Kaggle column names."""
    df = df.copy()

    df["transaction_hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["amount_log"] = np.log1p(df["Amount"])

    amount_std = df["Amount"].std()
    df["amount_zscore"] = (df["Amount"] - df["Amount"].mean()) / (
        amount_std if amount_std > 0 else 1.0
    )

    df["is_night"] = df["transaction_hour"].between(NIGHT_START, NIGHT_END).astype(int)

    high_threshold = df["Amount"].quantile(HIGH_AMOUNT_QUANTILE)
    df["is_high_amount"] = (df["Amount"] > high_threshold).astype(int)

    df = df.sort_values("Time")
    df["time_diff"] = df["Time"].diff().fillna(0)

    logger.info(
        "Engineered features added (high-amount threshold = %.2f)", high_threshold
    )
    return df


def build_dataset(csv_path: str) -> pd.DataFrame:
    """Full preprocessing pipeline: load → clean → engineer."""
    logger.info("Reading %s", csv_path)
    df = pd.read_csv(csv_path)
    df = clean(df)
    df = engineer_features(df)
    return df


def split_features_target(df: pd.DataFrame):
    """Return (X, y) using the shared MODEL_FEATURES contract."""
    missing = [c for c in MODEL_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing model features: {missing}")
    return df[MODEL_FEATURES], df[TARGET].astype(int)


if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    load_dotenv()
    path = os.getenv("DATA_PATH", "data/creditcard.csv")
    dataset = build_dataset(path)
    logger.info("Final dataset shape: %s", dataset.shape)
    logger.info("Fraud rate: %.4f%%", 100 * dataset[TARGET].mean())
