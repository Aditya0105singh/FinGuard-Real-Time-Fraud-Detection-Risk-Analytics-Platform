"""Tests for the analytics modules: statistical tests, threshold
optimization, and SHAP explanations."""

import numpy as np
import pandas as pd
import pytest

from src.predict import explain_one
from src.preprocess import MODEL_FEATURES
from src.stat_tests import (
    amount_mannwhitney,
    hourly_fraud_confints,
    night_vs_day_chi2,
    wilson_interval,
)
from src.threshold_optimizer import find_optimal, sweep_thresholds


def make_signal_frame(n: int = 6000) -> pd.DataFrame:
    """Synthetic frame with a deliberately strong night-fraud signal."""
    rng = np.random.default_rng(11)
    hours = rng.integers(0, 24, size=n)
    is_night = ((hours >= 0) & (hours <= 6)).astype(int)
    # Night transactions are 10x riskier by construction
    fraud_p = np.where(is_night == 1, 0.05, 0.005)
    fraud = (rng.random(n) < fraud_p).astype(int)
    # Fraud amounts drawn from a different distribution than legit
    amounts = np.where(
        fraud == 1,
        rng.exponential(scale=30, size=n),
        rng.exponential(scale=90, size=n),
    ).round(2)
    return pd.DataFrame({
        "transaction_hour": hours,
        "is_night": is_night,
        "Class": fraud,
        "Amount": amounts,
    })


class TestStatTests:
    def test_chi2_detects_planted_night_signal(self):
        result = night_vs_day_chi2(make_signal_frame())
        assert result["significant"] is True
        assert result["p_value"] < 0.05
        assert result["relative_risk"] > 2

    def test_mannwhitney_detects_planted_amount_difference(self):
        result = amount_mannwhitney(make_signal_frame())
        assert result["significant"] is True
        assert result["fraud_median_amount"] < result["legit_median_amount"]

    def test_wilson_interval_bounds(self):
        lo, hi = wilson_interval(5, 1000)
        assert 0.0 <= lo < 5 / 1000 < hi <= 1.0

    def test_wilson_interval_zero_n(self):
        assert wilson_interval(0, 0) == (0.0, 0.0)

    def test_wilson_never_negative_at_tiny_rates(self):
        lo, _ = wilson_interval(1, 100_000)
        assert lo >= 0.0

    def test_hourly_confints_cover_all_hours(self):
        ci = hourly_fraud_confints(make_signal_frame())
        assert set(ci["transaction_hour"]) == set(range(24))
        assert (ci["ci_low_pct"] <= ci["fraud_rate_pct"]).all()
        assert (ci["fraud_rate_pct"] <= ci["ci_high_pct"]).all()


class TestThresholdOptimizer:
    @pytest.fixture()
    def scored_data(self):
        """Well-separated synthetic scores: frauds high, legits low."""
        rng = np.random.default_rng(3)
        n = 2000
        y_true = (rng.random(n) < 0.05).astype(int)
        y_prob = np.clip(
            np.where(y_true == 1,
                     rng.normal(0.8, 0.15, n),
                     rng.normal(0.15, 0.1, n)),
            0, 1,
        )
        amounts = rng.exponential(scale=100, size=n)
        return y_true, y_prob, amounts

    def test_sweep_covers_full_range(self, scored_data):
        curve = sweep_thresholds(*scored_data, fp_cost=30.0)
        thresholds = [p["threshold"] for p in curve]
        assert min(thresholds) == 0.01
        assert max(thresholds) == 0.99
        assert len(curve) == 99

    def test_costs_are_consistent(self, scored_data):
        curve = sweep_thresholds(*scored_data, fp_cost=30.0)
        for point in curve:
            assert point["total_cost"] == pytest.approx(
                point["missed_fraud_cost"] + point["friction_cost"], abs=0.05
            )

    def test_recall_monotonically_decreases_with_threshold(self, scored_data):
        curve = sweep_thresholds(*scored_data, fp_cost=30.0)
        recalls = [p["recall"] for p in curve]
        assert all(a >= b for a, b in zip(recalls, recalls[1:]))

    def test_optimal_beats_or_matches_default(self, scored_data):
        curve = sweep_thresholds(*scored_data, fp_cost=30.0)
        optimal = find_optimal(curve)
        default = next(p for p in curve if p["threshold"] == 0.50)
        assert optimal["total_cost"] <= default["total_cost"]

    def test_extreme_thresholds_are_costly(self, scored_data):
        """At t=0.01 friction explodes; at t=0.99 missed fraud explodes."""
        curve = sweep_thresholds(*scored_data, fp_cost=30.0)
        optimal = find_optimal(curve)
        assert curve[0]["total_cost"] > optimal["total_cost"]
        assert curve[-1]["total_cost"] > optimal["total_cost"]


class TestExplainability:
    def test_explain_returns_top_factors(self, artifacts_dir):
        import joblib

        model = joblib.load(artifacts_dir / "model.pkl")
        scaler = joblib.load(artifacts_dir / "scaler.pkl")
        row = scaler.transform(np.zeros((1, len(MODEL_FEATURES))))
        factors = explain_one(model, row, top_k=5)

        assert 1 <= len(factors) <= 5
        for factor in factors:
            assert factor["feature"] in MODEL_FEATURES
            assert factor["direction"] in {"increases_risk", "decreases_risk"}
            assert isinstance(factor["contribution"], float)

    def test_factors_sorted_by_magnitude(self, artifacts_dir):
        import joblib

        model = joblib.load(artifacts_dir / "model.pkl")
        scaler = joblib.load(artifacts_dir / "scaler.pkl")
        row = scaler.transform(np.ones((1, len(MODEL_FEATURES))))
        factors = explain_one(model, row, top_k=5)
        magnitudes = [abs(f["contribution"]) for f in factors]
        assert magnitudes == sorted(magnitudes, reverse=True)
