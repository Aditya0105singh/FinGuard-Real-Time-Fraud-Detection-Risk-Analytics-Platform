"""FinGuard — Fraud Analytics Dashboard (Streamlit).

Pages:
  🏠 Home                — landing page with headline outcomes
  📊 Portfolio Overview  — KPIs and volume trends
  🔍 Fraud Analysis      — temporal & amount patterns
  🧩 Segments            — K-means behavioural segments
  📋 Executive Insights  — quantified findings + recommendations
  🤖 Model Performance   — metrics, cost curve, strategy bake-off
  ⚡ Live Prediction     — real-time scoring via FastAPI + SHAP

Run:
    streamlit run dashboard/app.py
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

# Ensure the dashboard directory is on the path so `theme` can be imported
# whether the app is run from the repo root (Streamlit Cloud) or locally.
_DASHBOARD_DIR = Path(__file__).parent
if str(_DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(_DASHBOARD_DIR))

from theme import (  # noqa: E402
    AMBER, BLUE, BLUE_LIGHT, CHART_SEQ, GREEN, GREEN_BG, GREEN_BORDER,
    INDIGO, NAVY, NAVY_2, ORANGE, RED, RISK_COLORS,
    SLATE_200, SLATE_400, SLATE_500, SLATE_600, SLATE_700, SLATE_900,
    feature_card, hero, inject_css, insight_card,
    register_plotly_template, risk_banner, section, step,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.dashboard")

# On Streamlit Cloud set API_BASE_URL in the app's Secrets manager.
# Locally it falls back to localhost.
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Resolve paths relative to the *repo root* (one level up from dashboard/)
_REPO_ROOT   = _DASHBOARD_DIR.parent
DATA_PATH    = os.getenv("DATA_PATH",    str(_REPO_ROOT / "data" / "creditcard.csv"))
ARTIFACT_DIR = os.getenv("ARTIFACT_DIR", str(_REPO_ROOT / "artifacts"))

st.set_page_config(
    page_title="FinGuard — Fraud Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()
register_plotly_template()


# ── helpers ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading transaction data…")
def load_data() -> pd.DataFrame:
    """Load transactions from PostgreSQL → CSV → synthetic demo (in that order)."""
    db = os.getenv("DATABASE_URL")
    if db:
        try:
            from sqlalchemy import create_engine
            engine = create_engine(db)
            df = pd.read_sql(
                'SELECT amount AS "Amount", class AS "Class", '
                'time_offset AS "Time", transaction_hour, day_of_week '
                "FROM transactions",
                engine,
            )
            logger.info("Loaded %d rows from PostgreSQL", len(df))
            return df
        except Exception as exc:
            logger.warning("PostgreSQL unavailable (%s) — CSV fallback", exc)
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df["transaction_hour"] = ((df["Time"] // 3600) % 24).astype(int)
        df["day_of_week"]      = ((df["Time"] // 86400) % 7).astype(int)
        logger.info("Loaded %d rows from CSV", len(df))
        return df
    # ── Synthetic demo data (Streamlit Cloud — no CSV available) ──────────
    logger.warning("CSV not found — generating synthetic demo dataset")
    rng = np.random.default_rng(42)
    n_legit, n_fraud = 28_400, 492
    n = n_legit + n_fraud
    labels = np.array([0] * n_legit + [1] * n_fraud)
    amounts = np.concatenate([
        rng.lognormal(mean=3.1, sigma=1.2, size=n_legit),   # legit: median ~€22
        rng.lognormal(mean=2.3, sigma=0.9, size=n_fraud),   # fraud: smaller median
    ])
    times = rng.uniform(0, 172_800, size=n)  # 48-hour window
    perm = rng.permutation(n)
    df = pd.DataFrame({
        "Amount": np.clip(amounts[perm], 0.01, 25_000),
        "Class":  labels[perm],
        "Time":   times[perm],
    })
    df["transaction_hour"] = ((df["Time"] // 3600) % 24).astype(int)
    df["day_of_week"]      = ((df["Time"] // 86400) % 7).astype(int)
    return df


@st.cache_data
def load_artifact(name: str) -> dict:
    path = os.path.join(ARTIFACT_DIR, name)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def artifact_img(filename: str, hint: str) -> None:
    path = os.path.join(ARTIFACT_DIR, filename)
    if os.path.exists(path):
        st.image(path, use_column_width=True)
    else:
        st.info(hint, icon="ℹ️")


_USING_SYNTHETIC = not os.path.exists(DATA_PATH) and not os.getenv("DATABASE_URL")


def need_data() -> "pd.DataFrame | None":
    """Return the transaction DataFrame, always succeeds (may be synthetic)."""
    df = load_data()
    if _USING_SYNTHETIC:
        st.info(
            "📊 **Demo mode** — showing a synthetic dataset that mirrors the real "
            "distribution (28,892 rows, 1.70% fraud). "
            "The full 284K-row `creditcard.csv` is not stored in the repo for size reasons.",
            icon="ℹ️",
        )
    return df


def _missing(msg: str) -> None:
    st.warning(msg, icon="⚙️")


# ── Page: Home ─────────────────────────────────────────────────────────
def page_home() -> None:
    metrics   = load_artifact("metrics.json")
    threshold = load_artifact("threshold_analysis.json")
    segments  = load_artifact("segments.json")

    _txn_chip  = "28,892 transactions (demo)" if _USING_SYNTHETIC else "284,807 transactions"
    _rate_chip = "1.70% fraud rate (demo)"   if _USING_SYNTHETIC else "0.17% fraud rate"
    hero(
        "🛡️ FinGuard",
        "Real-time credit card fraud detection that treats fraud as a "
        "business decision — statistically validated patterns, "
        "cost-optimised thresholds, and a SHAP explanation behind every score.",
        chips=[
            _txn_chip, _rate_chip, "XGBoost + MLflow",
            "FastAPI scoring", "Exact TreeSHAP", "PostgreSQL", "Docker + CI/CD",
        ],
    )

    recall  = metrics.get("recall")
    ap      = metrics.get("average_precision")
    savings = threshold.get("savings_vs_default_eur")
    seg     = (segments.get("segments") or [{}])[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Fraud caught (recall)",
        f"{recall:.1%}" if recall else "—",
        help="Share of true fraud detected on the held-out test set",
    )
    c2.metric(
        "Average precision",
        f"{ap:.3f}" if ap else "—",
        help="PR-AUC — the honest metric at a 0.17% base rate",
    )
    c3.metric(
        "Cost saved (threshold)",
        f"€{savings:,.0f}" if savings else "—",
        help="Expected savings vs the default 0.5 threshold",
    )
    c4.metric(
        "Riskiest segment",
        f"{seg.get('pct_of_all_fraud', 0):.0f}% of fraud"
        if seg.get("pct_of_all_fraud") else "—",
        delta=f"in {seg.get('pct_of_transactions', 0):.0f}% of volume"
        if seg.get("pct_of_transactions") else None,
        delta_color="off",
        help="K-means segment with highest fraud concentration",
    )

    st.divider()
    section(
        "The problem",
        "Card fraud costs the industry $32B+ a year — "
        "and every false decline burns customer goodwill too.",
    )
    if _USING_SYNTHETIC:
        st.markdown(
            "Only **492 of 28,892** transactions in this demo are fraudulent (1.70%). "
            "The real Kaggle dataset has 492 frauds in 284,807 rows (0.17%). "
            "A model that approves everything is *99.8% accurate* "
            "and catches **zero** fraud — so this project optimises what actually "
            "matters: euros lost to missed fraud vs euros burned on false declines."
        )
    else:
        st.markdown(
            "Only **492 of 284,807** transactions are fraudulent. "
            "A model that approves everything is *99.8% accurate* "
            "and catches **zero** fraud — so this project optimises what actually "
            "matters: euros lost to missed fraud vs euros burned on false declines."
        )

    st.divider()
    section("What makes this system different")
    col1, col2, col3 = st.columns(3)
    with col1:
        feature_card("🔬", "Statistically validated",
            "Every claimed pattern passes a formal test: chi-square for "
            "night-time risk, Mann-Whitney U for amounts, Wilson intervals "
            "on hourly rates.")
    with col2:
        feature_card("💶", "Cost-optimised decisions",
            "The decision threshold is swept against a real cost model "
            "(missed-fraud € vs false-decline friction €) — not defaulted to 0.5.")
    with col3:
        feature_card("🔍", "Explainable by design",
            "Every prediction ships with its top SHAP factors — because "
            '"the model said so" isn\'t an acceptable reason to decline a customer.')

    st.write("")
    col4, col5, col6 = st.columns(3)
    with col4:
        feature_card("⚖️", "Evidence over habit",
            "Four imbalance strategies (SMOTE, class weighting, "
            "undersampling, unsupervised) raced in MLflow — "
            "experiment picked the winner.")
    with col5:
        feature_card("🧩", "Behavioural segmentation",
            "K-means isolates a segment holding a majority of all fraud "
            "in a minority of volume — the cheapest place to add controls.")
    with col6:
        feature_card("🚀", "Production engineering",
            "Pydantic-validated FastAPI, 41 automated tests, "
            "Docker Compose, GitHub Actions CI/CD, MLflow model registry.")

    st.divider()
    section("Suggested path through the app")
    step(1, "Portfolio Overview",  "the raw numbers — volume, class balance, and time patterns.")
    step(2, "Fraud Analysis & Segments", "the statistical patterns and behavioural clusters.")
    step(3, "Executive Insights", "the findings memo — what the data says and what to do.")
    step(4, "Live Prediction",    "score a transaction and watch the model explain itself.")
    step(5, "Model Performance",  "cost curve, strategy bake-off, and honest metrics.")

    with st.expander("🏗️ Architecture overview"):
        st.code(
            "creditcard.csv ──► PostgreSQL (15 SQL analytics queries)\n"
            "       │\n"
            "       ▼\n"
            "preprocess ──► stat tests │ segmentation │ insights\n"
            "       │\n"
            "       ▼\n"
            "train (XGBoost) ──► MLflow runs + registry\n"
            "       │                 │\n"
            "       ▼                 ▼\n"
            "threshold optimizer   bake-off (4 strategies)\n"
            "       │\n"
            "       ▼\n"
            "artifacts/ ──► FastAPI /predict (+SHAP) ──► this dashboard",
            language="text",
        )


# ── Page: Portfolio Overview ───────────────────────────────────────────
def page_overview() -> None:
    df = need_data()
    if df is None:
        return

    section(
        "Portfolio Overview",
        "The shape of the problem: volume, fraud incidence, and exposure.",
    )

    total        = len(df)
    fraud_count  = int(df["Class"].sum())
    fraud_rate   = 100 * fraud_count / total
    fraud_amount = df.loc[df["Class"] == 1, "Amount"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total transactions", f"{total:,}")
    c2.metric("Fraud cases",        f"{fraud_count:,}")
    c3.metric("Fraud rate",         f"{fraud_rate:.3f}%",
              help="Extreme imbalance that drives every design decision")
    c4.metric("Fraud exposure",     f"€{fraud_amount:,.0f}",
              help="Total value of fraudulent transactions in the window")

    left, right = st.columns([3, 2], gap="large")

    with left:
        section("Transaction volume over time")
        df_t = df.copy()
        df_t["hour_bucket"] = (df_t["Time"] // 3600).astype(int)
        volume = df_t.groupby("hour_bucket").size().reset_index(name="transactions")
        fig = px.area(
            volume, x="hour_bucket", y="transactions",
            labels={"hour_bucket": "Hours since first transaction",
                    "transactions": "Transactions / hour"},
        )
        fig.update_traces(line_color=BLUE, fillcolor="rgba(37,99,235,0.1)")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Two daily cycles — the dataset covers ~48 hours of card activity. "
            "Overnight troughs are where fraud *rates* spike (see Fraud Analysis)."
        )

    with right:
        section("Class balance")
        fig = go.Figure(go.Pie(
            values=[total - fraud_count, fraud_count],
            labels=["Legitimate", "Fraud"],
            hole=0.6,
            marker=dict(colors=[GREEN, RED]),
            textinfo="label+percent",
            textfont=dict(size=13),
        ))
        fig.add_annotation(
            text=f"<b>{fraud_rate:.2f}%</b><br><span style='font-size:11px'>fraud</span>",
            showarrow=False, font_size=15,
        )
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Why accuracy is banned here: predicting 'legitimate' every time "
            "scores 99.8% and catches zero fraud."
        )


# ── Page: Fraud Analysis ───────────────────────────────────────────────
def page_fraud_analysis() -> None:
    df = need_data()
    if df is None:
        return

    section(
        "Fraud Analysis",
        "When and at what ticket size fraud happens — every pattern "
        "is backed by a formal statistical test (see Executive Insights).",
    )

    tab_hour, tab_heat, tab_amount = st.tabs(
        ["⏰  By hour", "🗓️  Hour × day heatmap", "💶  Amount distributions"]
    )

    with tab_hour:
        hourly = (
            df.groupby("transaction_hour")["Class"]
            .agg(["mean", "sum", "count"]).reset_index()
        )
        hourly["fraud_rate_pct"] = 100 * hourly["mean"]
        baseline = 100 * df["Class"].mean()
        fig = px.bar(
            hourly, x="transaction_hour", y="fraud_rate_pct",
            labels={"transaction_hour": "Hour of day",
                    "fraud_rate_pct": "Fraud rate (%)"},
            color="fraud_rate_pct",
            color_continuous_scale=["#DBEAFE", BLUE, RED],
        )
        fig.add_hline(y=baseline, line_dash="dash", line_color=SLATE_500,
                      annotation_text=f"baseline {baseline:.3f}%",
                      annotation_font_color=SLATE_500)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Night hours (00–06) run well above baseline — chi-square "
            "confirms a 3.25× relative risk (p ≈ 10⁻³²), justifying "
            "the model's `is_night` feature."
        )

    with tab_heat:
        pivot = (
            df[df["Class"] == 1]
            .groupby(["day_of_week", "transaction_hour"])
            .size().reset_index(name="frauds")
            .pivot(index="day_of_week", columns="transaction_hour", values="frauds")
            .fillna(0)
        )
        fig = px.imshow(
            pivot, aspect="auto",
            color_continuous_scale="Blues",
            labels=dict(x="Hour of day", y="Day", color="Fraud count"),
        )
        fig.update_layout(margin=dict(t=20))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Absolute fraud counts by hour and day of the capture window.")

    with tab_amount:
        sample_legit = df[df["Class"] == 0].sample(
            min(20_000, int((df["Class"] == 0).sum())), random_state=42
        )
        plot_df = pd.concat([sample_legit, df[df["Class"] == 1]])
        plot_df["Type"] = np.where(plot_df["Class"] == 1, "Fraud", "Legitimate")
        fig = px.histogram(
            plot_df[plot_df["Amount"] <= 500],
            x="Amount", color="Type", barmode="overlay",
            nbins=60, histnorm="percent", opacity=0.72,
            color_discrete_map={"Legitimate": GREEN, "Fraud": RED},
            labels={"Amount": "Transaction amount (€)", "percent": "Share (%)"},
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Counter-intuitive but Mann-Whitney-confirmed: fraud skews *smaller* "
            "(median €9.82 vs €22.00) — fraudsters probe with small amounts. "
            "Clipped at €500; 20K legitimate sample for readability."
        )


# ── Page: Segments ─────────────────────────────────────────────────────
def page_segments() -> None:
    segments = load_artifact("segments.json")

    section(
        "Behavioural Segments",
        "K-means clustering on amount, time-of-day, and PCA features — "
        "where in the portfolio does fraud concentrate?",
    )

    if not segments:
        _missing("No segmentation artifacts yet — run `python -m src.segmentation`.")
        return

    baseline = segments["baseline_fraud_rate_pct"]
    seg_df   = pd.DataFrame(segments["segments"])
    worst    = seg_df.iloc[0]

    c1, c2, c3 = st.columns(3)
    c1.metric("Segments",         segments["n_clusters"],
              help="k chosen for interpretability; silhouette-validated")
    c2.metric("Silhouette score", f"{segments['silhouette_score']:.3f}",
              help="Cluster separation on a 10K sample (higher = more distinct)")
    c3.metric("Baseline fraud",   f"{baseline:.3f}%")

    st.error(
        f"🎯 **Headline:** segment {int(worst['segment'])} holds just "
        f"**{worst['pct_of_transactions']:.1f}% of transactions** but "
        f"**{worst['pct_of_all_fraud']:.1f}% of all fraud** — "
        f"{worst['risk_multiple_vs_baseline']:.1f}× baseline risk, "
        f"€{worst['fraud_amount_eur']:,.0f} fraud value."
    )

    left, right = st.columns(2, gap="large")
    with left:
        section("Fraud rate by segment")
        fig = px.bar(
            seg_df, x="segment", y="fraud_rate_pct",
            labels={"fraud_rate_pct": "Fraud rate (%)", "segment": "Segment"},
            color="risk_multiple_vs_baseline",
            color_continuous_scale=["#DBEAFE", RED],
        )
        fig.add_hline(y=baseline, line_dash="dash", line_color=SLATE_500,
                      annotation_text="portfolio baseline",
                      annotation_font_color=SLATE_500)
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with right:
        section("Share of volume vs share of fraud")
        melted = seg_df.melt(
            id_vars="segment",
            value_vars=["pct_of_transactions", "pct_of_all_fraud"],
            var_name="metric", value_name="pct",
        )
        melted["metric"] = melted["metric"].map({
            "pct_of_transactions": "% of volume",
            "pct_of_all_fraud":    "% of all fraud",
        })
        fig = px.bar(
            melted, x="segment", y="pct", color="metric",
            barmode="group",
            color_discrete_map={"% of volume": BLUE, "% of all fraud": RED},
            labels={"pct": "Share (%)", "segment": "Segment"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📋 Full segment profiles"):
        st.dataframe(seg_df, use_container_width=True, hide_index=True)
    with st.expander("🗺️ Cluster scatter (hour × log-amount)"):
        artifact_img("segments.png", "Run `python -m src.segmentation` to generate.")


# ── Page: Executive Insights ───────────────────────────────────────────
def page_insights() -> None:
    insights = load_artifact("insights.json")

    section(
        "Executive Insights",
        "The analyst's memo: every finding is quantified and paired with an action.",
    )

    if not insights:
        _missing("No insights artifact yet — run `python -m src.insights`.")
        return

    st.caption(
        f"Generated {insights.get('generated_on', 'n/a')} · "
        f"{insights.get('n_findings', 0)} findings"
    )

    for i, finding in enumerate(insights.get("findings", []), start=1):
        insight_card(i, finding["title"], finding["finding"],
                     finding["recommendation"])


# ── Page: Model Performance ────────────────────────────────────────────
def page_model_performance() -> None:
    metrics = load_artifact("metrics.json")

    section(
        "Model Performance",
        "Held-out test metrics, cost-optimal operating point, and the "
        "evidence behind the imbalance-strategy choice.",
    )

    if not metrics:
        _missing("No training artifacts yet — run `python -m src.train` then `python -m src.evaluate`.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Recall",         f"{metrics.get('recall', 0):.1%}",
              help="Share of true fraud caught — each miss costs the full transaction amount")
    c2.metric("Precision",      f"{metrics.get('precision', 0):.1%}",
              help="Share of alerts that are real fraud")
    c3.metric("Avg Precision",  f"{metrics.get('average_precision', 0):.3f}",
              help="PR-AUC — the honest headline metric at 0.17% base rate")
    c4.metric("AUC-ROC",        f"{metrics.get('auc_roc', 0):.3f}")
    c5.metric("F1",             f"{metrics.get('f1', 0):.3f}")

    tab_cost, tab_bakeoff, tab_diag = st.tabs(
        ["💶  Cost threshold", "⚖️  Strategy bake-off", "📈  Diagnostics"]
    )

    with tab_cost:
        threshold = load_artifact("threshold_analysis.json")
        if threshold:
            opt = threshold["optimal"]
            t1, t2, t3, t4 = st.columns(4)
            t1.metric("Optimal threshold",     f"{opt['threshold']:.2f}",
                      help="Minimises missed-fraud € + false-decline friction €")
            t2.metric("Savings vs default 0.5",f"€{threshold['savings_vs_default_eur']:,.0f}")
            t3.metric("Recall @ optimal",      f"{100 * opt['recall']:.1f}%")
            t4.metric("FP cost assumption",    f"€{threshold['fp_cost_assumption_eur']:.0f}",
                      help="Configurable via FP_COST env var; industry range €15–118")

            curve = pd.DataFrame(threshold["curve"])
            fig = go.Figure()
            fig.add_scatter(x=curve["threshold"], y=curve["total_cost"],
                            name="Total cost",
                            line=dict(color=RED, width=3))
            fig.add_scatter(x=curve["threshold"], y=curve["missed_fraud_cost"],
                            name="Missed fraud",
                            line=dict(dash="dash", color=AMBER, width=2))
            fig.add_scatter(x=curve["threshold"], y=curve["friction_cost"],
                            name="False-decline friction",
                            line=dict(dash="dash", color=BLUE, width=2))
            fig.add_vline(x=opt["threshold"], line_color=GREEN, line_width=2,
                          annotation_text=f"optimal {opt['threshold']:.2f}",
                          annotation_font_color=GREEN)
            fig.add_vline(x=0.5, line_dash="dot", line_color=SLATE_500,
                          annotation_text="default 0.5",
                          annotation_font_color=SLATE_500)
            fig.update_layout(
                xaxis_title="Decision threshold",
                yaxis_title="Expected cost (EUR, test set)",
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "The 0.5 default assumes both error types cost the same — "
                "they never do in fraud. The optimizer converts an ML "
                "hyperparameter into a business decision."
            )
        else:
            st.info("Run `python -m src.threshold_optimizer` for the cost curve.", icon="⚙️")

    with tab_bakeoff:
        comparison = load_artifact("model_comparison.json")
        if comparison:
            best = comparison["best_by_average_precision"]
            st.success(
                f"🏆 **Best by average precision: {best}** — the experiment, "
                "not habit, picked the strategy.",
            )
            comp_df = (
                pd.DataFrame(comparison["results"])
                .T.reset_index()
                .rename(columns={"index": "strategy"})
            )
            st.dataframe(
                comp_df.round(4),
                use_container_width=True,
                hide_index=True,
                column_config={
                    "average_precision": st.column_config.ProgressColumn(
                        "avg precision", min_value=0, max_value=1, format="%.4f",
                    ),
                },
            )
            st.caption(
                "All four candidates trained on the identical stratified split, "
                "tracked as separate MLflow runs."
            )
        else:
            st.info("Run `python -m src.compare_models` for the bake-off.", icon="⚙️")

    with tab_diag:
        left, right = st.columns(2, gap="large")
        with left:
            section("Confusion matrix")
            cm = metrics.get("confusion_matrix")
            if cm:
                fig = go.Figure(go.Heatmap(
                    z=cm,
                    x=["Pred: Legit", "Pred: Fraud"],
                    y=["Actual: Legit", "Actual: Fraud"],
                    colorscale=[[0, "#DBEAFE"], [1, BLUE]],
                    text=cm, texttemplate="%{text:,}",
                    textfont=dict(size=14, color=SLATE_900),
                    showscale=False,
                ))
                fig.update_layout(margin=dict(t=16, b=16))
                st.plotly_chart(fig, use_container_width=True)
            else:
                artifact_img("confusion_matrix.png", "Run training first.")

            section("ROC curve")
            artifact_img("roc_curve.png", "Run `python -m src.evaluate`.")

        with right:
            section("Feature importance")
            artifact_img("feature_importance.png",
                         "Run training to generate the importance plot.")
            section("Precision-Recall curve")
            artifact_img("pr_curve.png", "Run `python -m src.evaluate`.")


# ── Page: Live Prediction ──────────────────────────────────────────────
PRESETS = {
    "🟢  Typical daytime purchase": {
        "amount": 49.90, "hour": 14, "zscore": -0.21,
        "v": [1.23, 0.18, 0.45, 0.62, -0.27, 0.35, 0.11, 0.06, 0.21, 0.04],
    },
    "🟡  Unusual night transaction": {
        "amount": 149.62, "hour": 2, "zscore": 0.24,
        "v": [-1.36, -0.07, 2.54, 1.38, -0.34, 0.46, 0.24, 0.10, 0.36, 0.09],
    },
    "🔴  Fraud-pattern probe": {
        "amount": 1.00, "hour": 3, "zscore": -0.35,
        "v": [-3.04, 3.16, -4.30, 4.73, -3.43, -1.64, -5.59, 1.40, -2.54, -4.74],
    },
}


def page_live_prediction() -> None:
    section(
        "Live Prediction",
        f"Scores against the FastAPI service at `{API_BASE_URL}` — "
        "every decision arrives with its SHAP explanation.",
    )

    preset_name = st.radio(
        "Quick scenario",
        list(PRESETS.keys()),
        horizontal=True,
        help="Pre-fills realistic feature patterns; tweak anything before scoring.",
    )
    preset = PRESETS[preset_name]

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        amount = c1.number_input("Amount (€)",           min_value=0.0,
                                  value=float(preset["amount"]),  step=1.0)
        hour   = c2.number_input("Transaction hour (0–23)", 0, 23,
                                  int(preset["hour"]))
        zscore = c3.number_input(
            "Amount z-score", value=float(preset["zscore"]), step=0.05,
            help="Standardised distance from the portfolio's typical amount",
        )

        is_night = 1 if 0 <= hour <= 6 else 0
        st.caption(
            f"`is_night` auto-derived: **{is_night}** "
            f"({'night window 00–06' if is_night else 'daytime'})"
        )

        st.markdown(
            "**PCA components V1–V10** — anonymised behavioural signals "
            "from the upstream feature pipeline"
        )
        v_cols  = st.columns(5)
        v_values = [
            v_cols[i % 5].number_input(
                f"V{i+1}", value=float(preset["v"][i]),
                step=0.1, key=f"v{i}_{preset_name}",
            )
            for i in range(10)
        ]
        submitted = st.form_submit_button(
            "⚡ Score transaction", type="primary", use_container_width=True
        )

    if not submitted:
        st.info(
            "Pick a scenario above and hit **Score transaction**. "
            "The response includes a fraud probability, risk tier, "
            "recommended action, and the top SHAP factors behind the decision.",
            icon="👆",
        )
        return

    payload = {
        "amount": amount, "transaction_hour": int(hour), "is_night": is_night,
        "amount_zscore": zscore, "v1_to_v10": v_values,
    }
    try:
        with st.spinner("Scoring… (first request may take ~30 s if the API is waking up on Render)"):
            resp = requests.post(f"{API_BASE_URL}/predict", json=payload, timeout=60)
            resp.raise_for_status()
            result = resp.json()
    except requests.exceptions.ConnectionError:
        st.warning(
            "⏳ **API is starting up** — Render free-tier services spin down after "
            "15 min of inactivity and need ~30 seconds to wake. "
            "Please wait a moment and click **Score transaction** again.",
            icon="🔌",
        )
        st.caption(f"Trying to reach: `{API_BASE_URL}/predict`")
        return
    except requests.exceptions.Timeout:
        st.warning(
            "⏳ **API timed out** — the service is likely cold-starting on Render. "
            "Wait 20–30 seconds and click **Score transaction** again.",
            icon="⌛",
        )
        return
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code in (502, 503, 504):
            st.warning(
                "🔄 **API is waking up (503 / 502)** — Render is restarting the "
                "container. Try again in ~30 seconds.",
                icon="☁️",
            )
        else:
            st.error(f"API returned an error: {exc}", icon="🔌")
        return
    except requests.RequestException as exc:
        st.error(f"API call failed: {exc}", icon="🔌")
        st.caption("If running locally: `uvicorn api.main:app --port 8000`")
        return

    risk  = result["risk_level"]
    color = RISK_COLORS.get(risk, SLATE_500)
    risk_banner(risk, result["recommendation"])

    gauge_col, shap_col = st.columns(2, gap="large")

    with gauge_col:
        m1, m2 = st.columns(2)
        m1.metric("Fraud probability", f"{result['fraud_probability']:.2%}")
        m2.metric("Model confidence",  f"{result['confidence']:.2%}",
                  help="Distance from the maximally uncertain 50/50 point")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=100 * result["fraud_probability"],
            number={"suffix": "%", "font": {"size": 38, "color": color}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": SLATE_400,
                         "tickfont": {"size": 11}},
                "bar": {"color": color, "thickness": 0.5},
                "bgcolor": "#F8FAFC",
                "bordercolor": SLATE_200,
                "steps": [
                    {"range": [0,  30], "color": "#D1FAE5"},
                    {"range": [30, 60], "color": "#FEF3C7"},
                    {"range": [60, 85], "color": "#FFEDD5"},
                    {"range": [85,100], "color": "#FEE2E2"},
                ],
                "threshold": {
                    "line": {"color": color, "width": 3},
                    "thickness": 0.8,
                    "value": 100 * result["fraud_probability"],
                },
            },
        ))
        fig.update_layout(height=260, margin=dict(t=20, b=10, l=20, r=20),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, use_container_width=True)

    with shap_col:
        section("Why this score?", "Exact TreeSHAP contributions in log-odds space.")
        factors = result.get("top_factors") or []
        if factors:
            shap_df = pd.DataFrame(factors)
            # Guard against anomalous SHAP values (>10 in log-odds is physically impossible)
            SHAP_LIMIT = 10.0
            anomalous = shap_df[shap_df["contribution"].abs() > SHAP_LIMIT]
            if not anomalous.empty:
                st.warning(
                    f"⚠️ SHAP value(s) for {list(anomalous['feature'])} appear anomalous "
                    f"(|value| > {SHAP_LIMIT}). This is a known model artifact — "
                    "those features are excluded from the chart.",
                    icon="⚠️",
                )
                shap_df = shap_df[shap_df["contribution"].abs() <= SHAP_LIMIT]
            shap_df = shap_df[::-1].reset_index(drop=True)
            if not shap_df.empty:
                max_abs = max(shap_df["contribution"].abs().max(), 0.5)
                fig = go.Figure(go.Bar(
                    x=shap_df["contribution"],
                    y=shap_df["feature"],
                    orientation="h",
                    marker=dict(
                        color=[RED if d == "increases_risk" else GREEN
                               for d in shap_df["direction"]],
                        line=dict(width=0),
                    ),
                    text=shap_df["contribution"].round(3),
                    textposition="outside",
                    textfont=dict(size=11, color=SLATE_700),
                ))
                fig.update_layout(
                    height=300,
                    xaxis_title="SHAP contribution (log-odds)",
                    xaxis_range=[-(max_abs * 1.4), max_abs * 1.4],
                    margin=dict(t=8, b=8),
                )
                st.plotly_chart(fig, use_container_width=True)
            st.caption("🔴 pushes score towards fraud · 🟢 pulls towards legitimate")
        else:
            st.info("The API did not return SHAP factors for this prediction.")

    st.divider()
    section("GenAI Explanation (Groq)", "Translates the ML features into a human-readable memo.")
    
    # Check if key is in Streamlit secrets / environment
    env_key = os.getenv("GROQ_API_KEY", "")
    
    if env_key:
        groq_api_key = env_key
    else:
        groq_api_key = st.text_input(
            "Enter Groq API Key to generate an explanation:",
            type="password",
            help="Free key at https://console.groq.com/keys — no credit card needed. "
                 "Leave blank to skip this section.",
        )
        st.caption("Get a free Groq API key at https://console.groq.com/keys (no credit card required).")
    
    if groq_api_key and factors:
        if st.button("Generate Explanation with Llama 3"):
            with st.spinner("Asking Groq..."):
                try:
                    from groq import Groq
                    client = Groq(api_key=groq_api_key)
                    
                    # Build prompt from SHAP factors
                    prompt = f"You are an expert fraud analyst explaining a decision to a non-technical manager. "
                    prompt += f"The transaction amount was €{amount}. The risk model flagged it with a {result['fraud_probability']:.1%} probability of fraud. "
                    prompt += f"The top factors driving this score are: "
                    for f in factors:
                        direction = "increased" if f["direction"] == "increases_risk" else "decreased"
                        prompt += f"\n- {f['feature']} ({direction} risk)"
                    prompt += "\n\nWrite a concise, 2-3 sentence business explanation of why this transaction was flagged or approved. Do not explain what SHAP or XGBoost is. Just explain the risk."
                    
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful, professional fraud analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama3-8b-8192",
                        temperature=0.3,
                        max_tokens=150,
                    )
                    explanation = chat_completion.choices[0].message.content
                    st.success(explanation)
                except Exception as e:
                    st.error(f"Failed to generate explanation: {e}")


# ── Router ─────────────────────────────────────────────────────────────
PAGES = {
    "🏠  Home":               page_home,
    "📊  Portfolio Overview":  page_overview,
    "🔍  Fraud Analysis":      page_fraud_analysis,
    "🧩  Segments":            page_segments,
    "📋  Executive Insights":  page_insights,
    "🤖  Model Performance":   page_model_performance,
    "⚡  Live Prediction":     page_live_prediction,
}


def main() -> None:
    with st.sidebar:
        st.markdown(
            f"<div style='padding:0.3rem 0 0.8rem 0;'>"
            f"<h1 style='color:#FFFFFF;font-size:1.45rem;font-weight:800;"
            f"margin:0 0 0.1rem 0;letter-spacing:-0.02em;'>🛡️ FinGuard</h1>"
            f"<p style='color:{SLATE_400};font-size:0.78rem;margin:0;"
            f"line-height:1.4;'>Real-Time Fraud Detection<br>"
            f"&amp; Risk Analytics</p>"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.divider()

        choice = st.radio(
            "Navigation",
            list(PAGES.keys()),
            label_visibility="collapsed",
        )

        st.divider()
        _sidebar_data = "28,892 rows (demo)" if _USING_SYNTHETIC else "284,807 transactions"
        st.markdown(
            f"<div style='font-size:0.72rem;color:{SLATE_400};line-height:1.7;'>"
            f"<strong style='color:{SLATE_400};'>API</strong> "
            f"<code style='background:rgba(255,255,255,0.08);color:#93C5FD;"
            f"padding:0.1rem 0.35rem;border-radius:4px;font-size:0.7rem;'>"
            f"{API_BASE_URL}</code><br>"
            f"<strong style='color:{SLATE_400};'>Model</strong> XGBoost v1.0 · TreeSHAP<br>"
            f"<strong style='color:{SLATE_400};'>Data</strong> {_sidebar_data}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Scroll to top on every page switch
    st.markdown(
        "<script>var m=window.parent.document.querySelector('.main');if(m)m.scrollTop=0;</script>",
        unsafe_allow_html=True,
    )
    PAGES[choice]()


if __name__ == "__main__":
    main()
