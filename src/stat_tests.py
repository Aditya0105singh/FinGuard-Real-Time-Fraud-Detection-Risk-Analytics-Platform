"""Statistical hypothesis tests backing the EDA and executive insights.

Three questions a fraud analyst must answer with evidence, not eyeballs:
  1. Is the night-time fraud elevation statistically significant?
       → chi-square test of independence (is_night × fraud)
  2. Do fraud and legitimate transactions differ in amount?
       → Mann-Whitney U (distributions are heavily non-normal, so a
         t-test's normality assumption would be violated)
  3. How uncertain are the per-hour fraud rates?
       → Wilson score confidence intervals (better than normal
         approximation at tiny proportions like 0.17%)
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.stat_tests")

ALPHA = 0.05


def night_vs_day_chi2(df: pd.DataFrame) -> dict:
    """Chi-square test: is fraud independent of night-time (00:00-06:59)?

    Expects columns `is_night` and `Class`. Returns test statistics plus
    the relative risk (night fraud rate / day fraud rate).
    """
    contingency = pd.crosstab(df["is_night"], df["Class"])
    chi2, p_value, dof, _ = stats.chi2_contingency(contingency)

    night_rate = df.loc[df["is_night"] == 1, "Class"].mean()
    day_rate = df.loc[df["is_night"] == 0, "Class"].mean()
    relative_risk = float(night_rate / day_rate) if day_rate > 0 else float("inf")

    result = {
        "test": "chi-square test of independence (is_night × fraud)",
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "night_fraud_rate_pct": float(100 * night_rate),
        "day_fraud_rate_pct": float(100 * day_rate),
        "relative_risk": relative_risk,
        "significant": bool(p_value < ALPHA),
    }
    logger.info(
        "Night vs day: RR=%.2fx, chi2=%.1f, p=%.2e (%s at alpha=%.2f)",
        relative_risk, chi2, p_value,
        "significant" if result["significant"] else "not significant", ALPHA,
    )
    return result


def amount_mannwhitney(df: pd.DataFrame) -> dict:
    """Mann-Whitney U: do fraud and legit amounts come from the same
    distribution? Chosen over a t-test because Amount is extremely
    right-skewed (normality assumption fails)."""
    fraud_amounts = df.loc[df["Class"] == 1, "Amount"]
    legit_amounts = df.loc[df["Class"] == 0, "Amount"]

    u_stat, p_value = stats.mannwhitneyu(
        fraud_amounts, legit_amounts, alternative="two-sided"
    )

    result = {
        "test": "Mann-Whitney U (fraud vs legitimate amounts)",
        "u_statistic": float(u_stat),
        "p_value": float(p_value),
        "fraud_median_amount": float(fraud_amounts.median()),
        "legit_median_amount": float(legit_amounts.median()),
        "fraud_mean_amount": float(fraud_amounts.mean()),
        "legit_mean_amount": float(legit_amounts.mean()),
        "significant": bool(p_value < ALPHA),
    }
    logger.info(
        "Amounts: fraud median=%.2f vs legit median=%.2f, U=%.0f, p=%.2e",
        result["fraud_median_amount"], result["legit_median_amount"],
        u_stat, p_value,
    )
    return result


def wilson_interval(successes: int, n: int, confidence: float = 0.95):
    """Wilson score interval for a binomial proportion.

    Preferred over the normal approximation when p is tiny (0.17%) —
    the normal interval can go negative and badly undercovers."""
    if n == 0:
        return 0.0, 0.0
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denom = 1 + z**2 / n
    centre = (p_hat + z**2 / (2 * n)) / denom
    margin = (z / denom) * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))
    # Pin the exact boundary cases — floating point can leave the lower
    # bound at ~1e-16 when successes == 0
    low = 0.0 if successes == 0 else max(0.0, centre - margin)
    high = 1.0 if successes == n else min(1.0, centre + margin)
    return low, high


def hourly_fraud_confints(df: pd.DataFrame, confidence: float = 0.95) -> pd.DataFrame:
    """Per-hour fraud rate with Wilson confidence intervals."""
    rows = []
    for hour, group in df.groupby("transaction_hour"):
        n = len(group)
        frauds = int(group["Class"].sum())
        lo, hi = wilson_interval(frauds, n, confidence)
        rows.append({
            "transaction_hour": int(hour),
            "txn_count": n,
            "fraud_count": frauds,
            "fraud_rate_pct": 100 * frauds / n,
            "ci_low_pct": 100 * lo,
            "ci_high_pct": 100 * hi,
        })
    return pd.DataFrame(rows).sort_values("transaction_hour").reset_index(drop=True)


def run_all(df: pd.DataFrame) -> dict:
    """Run the full battery; expects engineered columns from preprocess."""
    return {
        "night_vs_day": night_vs_day_chi2(df),
        "amount_distribution": amount_mannwhitney(df),
        "hourly_confints": hourly_fraud_confints(df).to_dict(orient="records"),
    }


if __name__ == "__main__":
    import json
    import os

    from dotenv import load_dotenv

    from src.preprocess import build_dataset

    load_dotenv()
    dataset = build_dataset(os.getenv("DATA_PATH", "data/creditcard.csv"))
    results = run_all(dataset)
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/stat_tests.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Results written to artifacts/stat_tests.json")
