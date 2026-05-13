"""Generate the executive insights memo (artifacts/insights.json).

Aggregates the statistical tests, threshold analysis, segmentation and
model metrics into 5–7 quantified findings, each paired with a concrete
business recommendation — the format an analytics team would present
to a fraud-operations manager.

Run after: train → evaluate → threshold_optimizer → segmentation.
(Findings that depend on missing artifacts are skipped gracefully.)

Usage:
    python -m src.insights
"""

import json
import logging
import os
import sys
from datetime import date

from dotenv import load_dotenv

from src.preprocess import build_dataset
from src.stat_tests import amount_mannwhitney, hourly_fraud_confints, night_vs_day_chi2

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.insights")

ARTIFACT_DIR = "artifacts"


def _load_artifact(name: str) -> dict | None:
    path = os.path.join(ARTIFACT_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    logger.warning("%s not found — related findings will be skipped", path)
    return None


def generate(csv_path: str) -> dict:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    df = build_dataset(csv_path)
    findings = []

    # 1. Night-time risk (chi-square)
    night = night_vs_day_chi2(df)
    findings.append({
        "id": "night_risk",
        "title": "Night transactions carry significantly elevated fraud risk",
        "finding": (
            f"Fraud rate between 00:00–06:59 is {night['night_fraud_rate_pct']:.3f}% "
            f"vs {night['day_fraud_rate_pct']:.3f}% during the day — a "
            f"{night['relative_risk']:.1f}x relative risk. Chi-square test confirms "
            f"this is not noise (χ²={night['chi2']:.0f}, p={night['p_value']:.1e})."
        ),
        "recommendation": (
            "Apply step-up authentication (OTP/3-D Secure) to night-time "
            "transactions above the median amount."
        ),
        "evidence": night,
    })

    # 2. Amount distribution difference (Mann-Whitney U)
    amounts = amount_mannwhitney(df)
    findings.append({
        "id": "amount_distribution",
        "title": "Fraudulent amounts follow a different distribution",
        "finding": (
            f"Median fraud amount is €{amounts['fraud_median_amount']:.2f} vs "
            f"€{amounts['legit_median_amount']:.2f} for legitimate transactions "
            f"(Mann-Whitney U, p={amounts['p_value']:.1e}). Fraudsters often probe "
            "with small amounts before larger attempts, so low-value anomalies "
            "matter as much as high-value ones."
        ),
        "recommendation": (
            "Do not filter alerts purely by amount; small-amount anomalies on "
            "new patterns deserve scoring weight."
        ),
        "evidence": amounts,
    })

    # 3. Peak fraud hours (with Wilson confidence intervals)
    hourly = hourly_fraud_confints(df)
    top = hourly.sort_values("fraud_rate_pct", ascending=False).head(3)
    peak_desc = ", ".join(
        f"{int(r.transaction_hour):02d}:00 ({r.fraud_rate_pct:.3f}% "
        f"[{r.ci_low_pct:.3f}–{r.ci_high_pct:.3f}%])"
        for r in top.itertuples()
    )
    findings.append({
        "id": "peak_hours",
        "title": "Fraud concentrates in identifiable peak hours",
        "finding": (
            f"The three riskiest hours (with 95% Wilson confidence intervals) are: "
            f"{peak_desc}. Overall portfolio baseline is "
            f"{100 * df['Class'].mean():.3f}%."
        ),
        "recommendation": (
            "Staff manual-review queues to match the hourly risk curve rather "
            "than uniform coverage."
        ),
        "evidence": {"top_hours": top.to_dict(orient="records")},
    })

    # 4. Cost-optimal threshold (if optimizer has run)
    threshold = _load_artifact("threshold_analysis.json")
    if threshold:
        opt = threshold["optimal"]
        findings.append({
            "id": "threshold",
            "title": "Cost-tuned decision threshold beats the 0.5 default",
            "finding": (
                f"Sweeping thresholds against a cost model (missed fraud = full "
                f"transaction amount, false decline = "
                f"€{threshold['fp_cost_assumption_eur']:.0f} friction) puts the "
                f"optimal operating point at t={opt['threshold']:.2f}, saving "
                f"€{threshold['savings_vs_default_eur']:,.0f} on the test window "
                f"vs the default 0.5 (recall {100 * opt['recall']:.1f}%, "
                f"precision {100 * opt['precision']:.1f}%)."
            ),
            "recommendation": (
                f"Deploy with threshold {opt['threshold']:.2f}; re-tune whenever "
                "the friction-cost assumption or fraud mix changes."
            ),
            "evidence": {
                "optimal": opt,
                "savings_vs_default_eur": threshold["savings_vs_default_eur"],
            },
        })

    # 5. Highest-risk segment (if segmentation has run)
    segments = _load_artifact("segments.json")
    if segments and segments.get("segments"):
        worst = segments["segments"][0]  # sorted by risk multiple
        findings.append({
            "id": "risky_segment",
            "title": "A small behavioural segment concentrates outsized fraud",
            "finding": (
                f"K-means segment {worst['segment']} holds only "
                f"{worst['pct_of_transactions']:.1f}% of transactions but "
                f"{worst['pct_of_all_fraud']:.1f}% of all fraud — "
                f"{worst['risk_multiple_vs_baseline']:.1f}x the baseline rate, "
                f"€{worst['fraud_amount_eur']:,.0f} of fraud value "
                f"(avg ticket €{worst['avg_amount']:.2f}, "
                f"{worst['night_share_pct']:.0f}% at night)."
            ),
            "recommendation": (
                "Route this segment to a stricter scoring threshold or "
                "rules overlay; it is the cheapest place to buy recall."
            ),
            "evidence": worst,
        })

    # 6. Model performance summary (if training has run)
    metrics = _load_artifact("metrics.json")
    if metrics and "recall" in metrics:
        findings.append({
            "id": "model_performance",
            "title": "Model performance on the held-out test set",
            "finding": (
                f"XGBoost with SMOTE achieves recall "
                f"{100 * metrics['recall']:.1f}%, precision "
                f"{100 * metrics['precision']:.1f}%, AUC-ROC "
                f"{metrics['auc_roc']:.3f} and average precision "
                f"{metrics['average_precision']:.3f} on 20% held-out data. "
                "With a 0.17% fraud base rate, average precision is the "
                "honest headline metric — accuracy is uninformative."
            ),
            "recommendation": (
                "Track precision/recall at the deployed threshold weekly; "
                "alert if recall drops more than 5 points."
            ),
            "evidence": {k: metrics[k] for k in
                         ("precision", "recall", "f1", "auc_roc",
                          "average_precision") if k in metrics},
        })

    payload = {
        "generated_on": date.today().isoformat(),
        "n_findings": len(findings),
        "findings": findings,
    }
    with open(os.path.join(ARTIFACT_DIR, "insights.json"), "w") as f:
        json.dump(payload, f, indent=2)
    logger.info("Wrote %d findings to artifacts/insights.json", len(findings))
    return payload


if __name__ == "__main__":
    try:
        generate(os.getenv("DATA_PATH", "data/creditcard.csv"))
    except Exception:
        logger.exception("Insight generation failed")
        sys.exit(1)
