"""Unit tests for preprocessing and inference logic."""

import numpy as np
import pandas as pd
import pytest

from src.predict import (
    build_feature_vector,
    predict_one,
    risk_from_probability,
)
from src.preprocess import MODEL_FEATURES, clean, engineer_features


def make_raw_frame(n: int = 200) -> pd.DataFrame:
    """Synthetic frame matching the Kaggle schema."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame({f"V{i}": rng.normal(size=n) for i in range(1, 29)})
    df["Time"] = np.sort(rng.integers(0, 172_800, size=n))
    df["Amount"] = rng.exponential(scale=80, size=n).round(2)
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    return df


class TestPreprocess:
    def test_clean_drops_duplicates(self):
        df = make_raw_frame(50)
        df_dup = pd.concat([df, df.head(10)])
        assert len(clean(df_dup)) == 50

    def test_engineered_columns_exist(self):
        df = engineer_features(make_raw_frame())
        for col in ("transaction_hour", "amount_log", "amount_zscore",
                    "is_night", "is_high_amount", "time_diff"):
            assert col in df.columns, f"missing engineered feature: {col}"

    def test_transaction_hour_range(self):
        df = engineer_features(make_raw_frame())
        assert df["transaction_hour"].between(0, 23).all()

    def test_is_night_definition(self):
        df = engineer_features(make_raw_frame())
        night = df[df["is_night"] == 1]["transaction_hour"]
        assert night.between(0, 6).all()

    def test_amount_log_is_log1p(self):
        df = engineer_features(make_raw_frame())
        np.testing.assert_allclose(df["amount_log"], np.log1p(df["Amount"]))

    def test_time_diff_non_negative_when_sorted(self):
        df = engineer_features(make_raw_frame())
        assert (df["time_diff"] >= 0).all()

    def test_model_features_present_after_engineering(self):
        df = engineer_features(make_raw_frame())
        assert all(col in df.columns for col in MODEL_FEATURES)


class TestRiskMapping:
    @pytest.mark.parametrize(
        "probability,expected",
        [
            (0.05, "LOW"),
            (0.29, "LOW"),
            (0.45, "MEDIUM"),
            (0.70, "HIGH"),
            (0.90, "CRITICAL"),
            (0.999, "CRITICAL"),
        ],
    )
    def test_risk_bands(self, probability, expected):
        level, recommendation = risk_from_probability(probability)
        assert level == expected
        assert recommendation


class TestInference:
    def test_feature_vector_order_matches_contract(self):
        vec = build_feature_vector(
            amount=100.0, transaction_hour=3, is_night=1,
            amount_zscore=0.5, v1_to_v10=list(range(10)),
        )
        assert vec.shape == (1, len(MODEL_FEATURES))
        # V1..V10 first, then Amount, hour, is_night, zscore
        assert list(vec[0][:10]) == list(range(10))
        assert vec[0][10] == 100.0
        assert vec[0][11] == 3
        assert vec[0][12] == 1
        assert vec[0][13] == 0.5

    def test_feature_vector_rejects_wrong_length(self):
        with pytest.raises(ValueError):
            build_feature_vector(100.0, 3, 1, 0.5, [0.0] * 7)

    def test_predict_one_end_to_end(self, artifacts_dir):
        import joblib

        model = joblib.load(artifacts_dir / "model.pkl")
        scaler = joblib.load(artifacts_dir / "scaler.pkl")
        result = predict_one(
            model, scaler,
            amount=149.62, transaction_hour=2, is_night=1,
            amount_zscore=0.24,
            v1_to_v10=[-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09],
        )
        assert 0.0 <= result["fraud_probability"] <= 1.0
        assert result["risk_level"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["recommendation"]
