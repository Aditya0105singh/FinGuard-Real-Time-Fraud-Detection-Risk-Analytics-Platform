# FinGuard — Interview Preparation Guide

> Based on full codebase analysis. Every fact, metric, and code reference is
> grounded in the actual implementation in this repository.
> Use this as your single reference before any interview, demo, or portfolio review.

---

## 1. Elevator Pitch (30–60 Seconds)

> *Use this with HR screeners, networking events, and LinkedIn messages.*

"I built FinGuard — a production-grade credit card fraud detection system. The core challenge is that only 0.17% of transactions are fraud, so a model that approves everything is 99.8% accurate and still catches zero fraud. I solved this by combining XGBoost with SMOTE oversampling, statistical hypothesis tests to validate every claimed pattern, and a custom cost model that finds the optimal decision threshold instead of blindly using 0.5. The result is a system that catches nearly 79% of fraud on held-out data, with a FastAPI scoring service that explains every decision using SHAP, a real-time Streamlit dashboard, and 41 automated tests. What sets it apart is the business framing — every ML decision is expressed in euros, not accuracy percentages."

---

## 2. Interview Explanation (3–5 Minutes)

> *Use this in technical phone screens and first-round interviews.*

**The problem:**
Credit card fraud is an extreme class imbalance problem. Out of 284,807 real transactions in this dataset, only 492 — 0.17% — are fraudulent. Any naive model that approves every transaction achieves 99.8% accuracy and catches zero fraud. That means accuracy is a completely meaningless metric here, and the entire system needs to be designed around what actually matters: money.

**Why it matters:**
Fraud costs the global payments industry $32 billion a year. But there are two types of errors — missing fraud and false declines — and they have completely different costs. Blocking a legitimate transaction costs goodwill and customer churn, estimated at €15–118 per incident in the industry. Letting through fraud costs the full transaction amount. A production system needs to price both explicitly.

**My approach:**
I built a complete ML pipeline from raw CSV to a production API. The data pipeline cleans the dataset, engineers four features — transaction hour, whether it's overnight, the amount's z-score, and the log-transformed amount — then trains an XGBoost classifier. I applied SMOTE oversampling strictly after the train/test split to prevent synthetic data from leaking into evaluation. I then built a cost model that sweeps every threshold from 0.01 to 0.99, calculates total expected cost at each point using the real transaction amounts, and picks the minimum — which turned out to be 0.86, saving €4,489 compared to the default 0.5.

**Architecture:**
The system has four layers. A data pipeline that writes artifacts. A FastAPI service with five endpoints — health, predict, batch-predict, stats, and docs — that loads the model once at startup using an LRU cache and returns SHAP explanations with every single prediction. A Streamlit dashboard with seven pages covering everything from portfolio KPIs to a live scoring interface. And a PostgreSQL database with 15 analytics queries for the BI layer.

**Key technical decisions:**
XGBoost because `tree_method='hist'` handles 283K rows fast and TreeSHAP is exact for tree models. Pydantic v2 for API validation — it catches wrong PCA vector length before the model even sees the data. MLflow for experiment tracking — I ran a four-way bake-off of SMOTE, class weighting, undersampling, and Isolation Forest, and class weighting actually won on average precision at 0.759.

**Outcome:**
AUC-ROC of 0.967, recall of 78.9%, average precision of 0.746 on held-out data. The SHAP explanations make every decision auditable — in fintech, "the model said so" is not an acceptable reason to block a customer. 41 automated tests covering API contracts, model inference, statistical tests, and the threshold optimizer, with no real dataset needed.

---

## 3. Deep Technical Walkthrough (10–15 Minutes)

> *Use this in senior engineer interviews and system design discussions.*

### 3.1 System Architecture

```
┌───────────────────────────────────────────────────────────────┐
│                      Data Layer                               │
│  creditcard.csv ──► src/ingest.py ──► PostgreSQL              │
│                           │                                   │
│                    src/preprocess.py                          │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                     ML Pipeline                               │
│  train → evaluate → threshold_optimizer → compare_models     │
│  segmentation → stat_tests → insights                         │
│  [All write to artifacts/]                                    │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                   Serving Layer                               │
│  api/main.py (FastAPI)  ←── artifacts/                        │
│  GET /    GET /health    POST /predict                        │
│  POST /batch-predict     GET /stats                           │
└───────────────────────────┬───────────────────────────────────┘
                            │
┌───────────────────────────▼───────────────────────────────────┐
│                 Presentation Layer                            │
│  dashboard/app.py (Streamlit) — 7 pages                       │
│  Reads: artifacts/ + API + PostgreSQL                         │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 Data Flow — Single Prediction Request

```
Dashboard (Streamlit)
  │
  │  POST /predict {amount, hour, is_night, zscore, v1_to_v10[10]}
  ▼
FastAPI (api/main.py)
  │ Pydantic validates: amount≥0, hour 0-23, len(v)==10
  │ _bundle_or_503() → get_bundle() [lru_cache, loaded once]
  │ build_feature_vector → scaler.transform → predict_proba
  │ risk_bands lookup → risk_level + recommendation
  │ booster.predict(DMatrix, pred_contribs=True) → SHAP values
  ▼
Response: {fraud_probability, risk_level, confidence,
           recommendation, top_factors[5]}
```

### 3.3 Training Pipeline — Critical Design Decisions

**Step 1 — Feature Engineering (`src/preprocess.py`)**

The 14 model features are:
```
V1, V2, ..., V10  ← PCA-anonymised behavioural signals (pre-anonymised in dataset)
Amount            ← raw transaction value in EUR
transaction_hour  ← (Time // 3600) % 24
is_night          ← 1 if hour in [0, 6] else 0
amount_zscore     ← (Amount - mean) / std, computed per batch
```

Four additional engineered features (`amount_log`, `is_high_amount`, `time_diff`,
`day_of_week`) are created but excluded from MODEL_FEATURES. They feed SQL analytics
and segmentation only.

**Step 2 — Split before everything else**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```

`stratify=y` is critical — with only 492 fraud cases, a random split could give
the test set as few as 80 fraud examples, making evaluation unreliable.

**Step 3 — Scale on train, apply to test**

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # transform only, never fit
```

`fit` only on training data. The test set statistics are never seen during fitting
— this is a common leakage vector that this design explicitly prevents.

**Step 4 — SMOTE strictly after split**

```python
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_scaled, y_train)
```

SMOTE synthesises new minority samples by interpolating between real ones. If
applied before the split, synthetic versions of test-set neighbours would appear
in training — a form of data leakage. Result: 226,980 training rows → 453,204
(50/50 balance).

**Step 5 — XGBoost with `eval_metric='aucpr'`**

Using PR-AUC as the internal eval metric rather than accuracy or AUC-ROC directly
aligns the training signal with the correct evaluation criterion for imbalanced
classification.

### 3.4 Cost-Based Threshold Optimisation (`src/threshold_optimizer.py`)

The core business insight: the 0.5 decision threshold is statistically arbitrary.
The real question is "at what probability does blocking a transaction save more
than it costs?"

**Cost model:**
```
Cost(threshold) = sum(amount_i for missed fraud transactions)
               + FP_COST × count(legitimate transactions flagged as fraud)
```

This is computed on the held-out test set using real transaction amounts (persisted
in `artifacts/test_set.npz`). FP_COST defaults to €30 (configurable via env var).
The optimal threshold at 0.86 reduces total cost from €11,670 to €7,181 — a 38.5%
saving.

### 3.5 Statistical Validation Layer (`src/stat_tests.py`)

Three tests, each chosen for a specific reason:

| Test | Question | Why this test |
|------|----------|---------------|
| Chi-square of independence | Is night-time correlated with fraud? | Categorical × categorical; no distribution assumption |
| Mann-Whitney U | Do fraud and legit amounts differ? | Amount is heavily right-skewed — t-test's normality assumption fails |
| Wilson score CI | How uncertain are per-hour fraud rates? | Normal approximation goes negative at p=0.17%; Wilson is always valid |

Results: night fraud rate is 3.25× day rate (χ²=141.3, p=1.38×10⁻³²); fraud
median €9.82 vs €22.00 (p=2.69×10⁻⁵).

### 3.6 SHAP Explanations (`src/predict.py`)

```python
booster = model.get_booster()
contribs = booster.predict(xgb.DMatrix(scaled_features), pred_contribs=True)[0]
```

This uses XGBoost's native TreeSHAP implementation via `pred_contribs=True`. Key
properties:
- **Exact, not approximate** — TreeSHAP computes true Shapley values for tree
  models in O(TLD) time where T=trees, L=leaves, D=depth
- **Log-odds space** — contributions sum to the raw log-odds score, not the
  probability
- **Last element is bias** — the 15th value (index 14) is the expected model
  output, excluded from feature contributions
- Top-5 contributions sorted by absolute magnitude are returned

### 3.7 API Layer (`api/main.py`)

**Model loading — lazy singleton:**
```python
@lru_cache(maxsize=1)
def get_bundle() -> ModelBundle:
    return ModelBundle()
```

The `@lru_cache(maxsize=1)` pattern means `joblib.load()` is called exactly once
per process regardless of request volume. The test suite calls
`model_loader.get_bundle.cache_clear()` before each test session to ensure a
fresh load against the temp artifacts.

**Validation before inference:**
```python
@field_validator("v1_to_v10")
@classmethod
def must_have_ten_components(cls, v):
    if len(v) != 10:
        raise ValueError(...)
```

This fires before the feature vector is assembled, returning a 422 instead of an
assertion error deep in numpy.

**Error hierarchy:**
- 422 — invalid input (Pydantic catches before model)
- 503 — model artifacts not found (FileNotFoundError from model_loader)
- 500 — unexpected inference failure

### 3.8 Test Architecture (`tests/`)

The test suite runs against zero real data. `conftest.py` trains a 10-estimator
XGBoost on 500 synthetic rows with a planted signal (`X[:,0] > 1.2` as fraud),
persists the same artifact set that `src/train.py` produces, and wires the
FastAPI TestClient to that temp directory via env vars. This means:

- CI runs in seconds with no dataset dependency
- The API contract tests exercise the full request-response cycle
- The `artifacts_dir` fixture is session-scoped — trained once, shared across
  all 41 tests

---

## 4. Project Architecture Breakdown

### Why This Architecture?

**Decoupled training and serving.** Training writes to `artifacts/`. The API only
reads from `artifacts/`. This means you can retrain without restarting the API
server, and the API never touches the raw dataset or the training code. This is a
standard MLOps pattern.

**Stateless API.** The model is loaded once at startup via `@lru_cache`. Every
request is independent. This makes horizontal scaling trivial — add more API
containers behind a load balancer with no shared state.

**Artifact-first pipeline.** Each pipeline stage writes a self-contained JSON
artifact. This means any downstream consumer (dashboard, insights generator,
threshold optimizer) can be run independently and rerun without re-executing
earlier stages.

### Technology Choices

| Layer | Technology | Chosen because |
|-------|-----------|----------------|
| ML model | XGBoost | Exact TreeSHAP, hist method for 283K rows, proven tabular baseline |
| API framework | FastAPI | Automatic Pydantic validation, OpenAPI docs, async-capable |
| Schema validation | Pydantic v2 | Catches malformed inputs (wrong vector length) before inference |
| Dashboard | Streamlit | Rapid analytics UI, Python-native, integrates directly with model artifacts |
| Experiment tracking | MLflow | Comparable multi-run bake-off, model registry, artifact versioning |
| Database | PostgreSQL | Window functions and CTEs for the 15 analytics queries |
| Containerisation | Docker Compose | Single command for 4-service stack (postgres, mlflow, api, dashboard) |
| Testing | pytest + TestClient | Session-scoped fixtures, no real dataset needed |

### Alternative Approaches Considered

| Decision | Chosen | Alternative | Why not chosen |
|----------|--------|-------------|----------------|
| Oversampling strategy | SMOTE (production) | scale_pos_weight | SMOTE has higher recall (78.9% vs 76.8%); bake-off showed both are valid |
| Threshold selection | Cost-sweep optimiser | F1-optimal threshold | F1 doesn't use real transaction amounts; cost model is directly interpretable |
| SHAP implementation | XGBoost native pred_contribs | SHAP library KernelExplainer | Native is exact and has no extra dependency |
| MLflow backend | Local mlruns/ (dev) | PostgreSQL-backed server | Local is zero-infra; env var swap for production |

---

## 5. Technical Decision Log

### Decision 1: XGBoost over Logistic Regression or Neural Network

**Chosen:** XGBoost with `tree_method='hist'`

**Why:** Three reasons specific to this use case. First, `tree_method='hist'` uses
histogram-based gradient approximation — handles 283K rows in seconds. Second,
native TreeSHAP support via `get_booster().predict(pred_contribs=True)` — exact
explanations, no approximation needed. Third, XGBoost with proper hyperparameters
typically matches deep learning on structured tabular data.

**Production verdict:** Yes. XGBoost and LightGBM dominate tabular fraud
leaderboards. A neural network would need far more tuning for marginal gain on
this feature set.

### Decision 2: SMOTE Applied Strictly After the Train/Test Split

**Chosen:** SMOTE for the production model

**Why:** Class imbalance prevention is required. SMOTE placement is critical:
applied after the split so synthetic samples derived from test-set neighbours
cannot appear in training. If applied before the split, the model has
effectively seen the test set through the interpolated points — a subtle but
invalidating form of data leakage.

**The counterpoint:** The bake-off showed `scale_pos_weight` actually beat SMOTE
on average precision (0.759 vs 0.746) and precision (52.5% vs 24.6%). The
production model uses SMOTE because recall is slightly higher (78.9% vs 76.8%).
The choice is documented and backed by experimental evidence.

### Decision 3: Cost-Based Threshold (0.86) over Default (0.5)

**Why not 0.5:** The 0.5 threshold assumes missing fraud and false declines cost
the same. At 0.5, total test-window cost is €11,670. At 0.86, it's €7,181.

**Trade-off:** Higher threshold means fewer false positives but more missed fraud.
The break-even depends on FP_COST assumption (€30 default). This is exposed as a
configurable environment variable so ops teams can calibrate it from real
chargeback data.

### Decision 4: TreeSHAP over LIME or Post-hoc Explanations

**Why XGBoost native `pred_contribs=True`:**
- Exact Shapley values (not approximate like LIME)
- O(TLD) complexity — fast enough for real-time API responses at sub-millisecond
- No separate library dependency (uses XGBoost's own booster)
- Consistent: same input always produces same explanation
- Contributions in log-odds space sum exactly to the prediction minus the bias term

### Decision 5: MLflow Local File Store for Development

**Why `MLFLOW_TRACKING_URI=mlruns`:** No infrastructure dependency for
development and demo. The Docker Compose stack includes a proper MLflow server
for the containerised environment. The code change for production is a single
env var.

### Decision 6: Pydantic v2 for API Validation

**Why:** The `v1_to_v10` field must contain exactly 10 values — fewer silently
produces wrong features, more silently gets truncated. The validator enforces
this contract at the API boundary, returning a 422 before any model code runs.
Added `ConfigDict(protected_namespaces=())` to response schemas to handle
Pydantic v2's `model_` namespace clash.

### Decision 7: Wilson Score Confidence Intervals

**Why Wilson over normal approximation:** The normal approximation interval
`p ± z√(p(1-p)/n)` can produce negative lower bounds when p is tiny (0.17%).
The Wilson interval is always in [0,1] and has better coverage at extreme
proportions. The rationale is documented in the code.

---

## 6. Ownership Analysis

Use these phrases in interviews to demonstrate genuine ownership:

**"I implemented this part":**
- "I implemented the cost-based threshold optimiser that sweeps 99 decision
  points and minimises a real euro cost function using the actual transaction
  amounts from the held-out test set."
- "I implemented the SHAP explanation layer using XGBoost's native TreeSHAP
  `pred_contribs=True` API, which gives exact Shapley values without any
  approximation."
- "I implemented the Wilson score confidence interval function for per-hour
  fraud rates, specifically because the normal approximation produces negative
  bounds at the 0.17% fraud rate."

**"I designed this workflow":**
- "I designed the training pipeline to apply SMOTE strictly after the train/test
  split to prevent synthetic samples from leaking into the evaluation set."
- "I designed the artifact-first architecture where training and serving are fully
  decoupled — the API reads from `artifacts/` and never touches training code."
- "I designed the four-way imbalance strategy bake-off so the SMOTE choice is
  backed by evidence rather than convention."

**"I solved this challenge":**
- "I solved the MLflow connectivity issue by switching from
  `http://localhost:5000` to the local `mlruns` file store, so the pipeline
  works without a running MLflow server."
- "I solved the Pydantic `model_version` namespace conflict by adding
  `model_config = ConfigDict(protected_namespaces=())` to the response schemas."
- "I solved the metric evaluation problem — using average precision instead of
  accuracy — by recognising that at 0.17% fraud rate, the 99.8%-accurate
  all-negative baseline catches nothing."

---

## 7. Challenges and Problem-Solving Stories (STAR Format)

### Challenge 1: Data Leakage from SMOTE

**Situation:** Needed to handle severe class imbalance (492 fraud cases in
284,807 transactions — 0.17%).

**Task:** Apply SMOTE oversampling without contaminating the evaluation set.

**Action:** Applied SMOTE strictly after the stratified train/test split and
after fitting the StandardScaler. The order is: (1) split, (2) fit scaler on
train only, (3) transform both sets, (4) apply SMOTE only to training data.
This is enforced structurally in the code — `smote.fit_resample` never sees
`X_test_scaled`.

**Result:** A clean evaluation: the model's 78.9% recall and 0.967 AUC-ROC
reflect performance on genuinely unseen real transactions, not synthetic ones.

**Interview line:** "SMOTE creates synthetic fraud samples by interpolating
between real ones. If you apply it before the split, you end up with synthetic
versions of test-set neighbours in your training data — the model has
effectively seen the test set. I solved this by making the split the first
operation and treating SMOTE as training-only augmentation."

---

### Challenge 2: The Default 0.5 Threshold is Wrong for Fraud

**Situation:** After training, the model had 78.9% recall at the default 0.5
threshold, but precision was only 24.6% — 75% of flagged transactions were
legitimate.

**Task:** Find a decision threshold that reflects actual business costs, not
statistical convention.

**Action:** Built a threshold sweep from 0.01 to 0.99, computing
`total_cost = missed_fraud_cost + friction_cost` at each point. Used the real
transaction amounts from the held-out test set as the cost of missed fraud, and
€30 per false decline as friction cost (configurable via env var).

**Result:** The optimal threshold is 0.86, reducing total expected cost from
€11,670 to €7,181 — a 38.5% saving. This also frames the ML problem in terms
a CFO understands: euros, not F1 scores.

**Interview line:** "The 0.5 threshold is a statistics convention, not a business
decision. I built a cost function that converts every threshold into an expected
euro cost on real data. The optimizer finds the minimum. The key insight is that
the FP cost and the fraud amount are completely asymmetric — a €1,000 fraud
caught is worth far more than avoiding ten €30 false declines."

---

### Challenge 3: Choosing the Right Test for Non-Normal Distributions

**Situation:** Needed to test whether fraudulent transaction amounts differ from
legitimate ones.

**Task:** Select a statistical test that is valid for the actual data
distribution.

**Action:** The transaction amount variable is heavily right-skewed. A t-test
assumes normality — it would be statistically invalid here. I used the
Mann-Whitney U test, which tests whether two distributions differ without any
distributional assumption. The reasoning is documented directly in the code.

**Result:** p=2.69×10⁻⁵ confirms the difference is real. The finding: fraud
skews *smaller* (median €9.82 vs €22.00) — fraudsters probe with small amounts,
which means amount-based filters would miss them entirely.

**Interview line:** "I could have used a t-test and gotten a significant result —
the distributions are different enough. But the t-test's normality assumption
fails for transaction amounts. Using the wrong test means your p-value is
theoretically invalid. The Mann-Whitney U test is the correct choice here, and
I documented that reasoning in the code."

---

### Challenge 4: Making Tests Independent of the Real Dataset

**Situation:** The test suite needed to cover the full API request cycle and
model inference, but the 143MB CSV dataset can't live in a repository.

**Task:** Write tests that exercise real code paths without any external data
dependency.

**Action:** The `conftest.py` `artifacts_dir` fixture trains a 10-estimator
XGBoost on 500 synthetic rows with a planted signal (`X[:,0] + noise > 1.2`
defines fraud). It persists `model.pkl`, `scaler.pkl`, and `metrics.json` to a
temp directory, then wires the FastAPI TestClient to those paths via env vars.
The `@lru_cache` is cleared before each session so the temp artifacts load fresh.

**Result:** 41 tests run in under 2 seconds, cover the full API contract, and
have zero external dependencies. CI works without the dataset.

**Interview line:** "Real ML test suites can't depend on 143MB datasets. I used
a session-scoped pytest fixture that trains a tiny model on synthetic data with a
guaranteed signal. The tests validate API contracts, response shapes, validation
errors, and inference correctness — not model quality. That's an important
distinction: these are software tests, not model evaluation."

---

## 8. Common Interview Questions and Answers

### Beginner Level

**Q: What is the fraud rate in your dataset?**
A: 492 out of 284,807 transactions — 0.173%. This extreme imbalance is the core
challenge. A model that approves everything is 99.83% accurate. That's why I use
average precision (PR-AUC) as the primary metric, not accuracy.

**Q: What features does your model use?**
A: Fourteen features: V1 through V10, which are pre-anonymised PCA components
from the dataset; Amount in euros; transaction_hour (0–23); is_night (binary
flag for hours 0–6); and amount_zscore (the standardised amount).

**Q: What is SMOTE?**
A: Synthetic Minority Oversampling Technique. It creates new fraud examples by
picking a real fraud transaction, finding its k nearest neighbours in feature
space, and interpolating between them. The critical implementation detail is
that it must be applied after the train/test split, never before — otherwise
synthetic versions of test examples leak into training.

**Q: What does SHAP stand for and how does it work here?**
A: SHapley Additive exPlanations. For tree models, XGBoost implements exact
TreeSHAP via `booster.predict(pred_contribs=True)`. This returns one contribution
per feature plus a bias term. Each value represents how much that feature pushed
the score above or below the model's expected output, in log-odds space. I return
the top 5 by absolute magnitude with each API response.

---

### Intermediate Level

**Q: Why did you pick XGBoost over a neural network?**
A: Three reasons. First, `tree_method='hist'` makes it fast on tabular data at
this scale. Second, TreeSHAP gives exact Shapley values — neural network
explanations are always approximations. Third, XGBoost with proper
hyperparameters typically matches deep learning on structured tabular data.

**Q: Why is AUC-ROC not your headline metric?**
A: AUC-ROC is optimistic under heavy imbalance because it includes the true
negative rate, which is inflated when 99.83% of transactions are legitimate.
Average precision (PR-AUC) only considers the precision-recall trade-off across
the positive class. My model has AUC-ROC of 0.967 and average precision of
0.746 — both are good, but average precision is the honest number at this fraud
rate.

**Q: How does your API handle the case where the model files don't exist?**
A: `model_loader.py` raises `FileNotFoundError` if either `model.pkl` or
`scaler.pkl` is missing. The `_bundle_or_503()` helper catches this and returns
an HTTP 503 with the message "Model artifacts not available. Train the model
first." A missing model is an infrastructure problem, not a client error.

**Q: Why does your test suite train its own model instead of using the
production model?**
A: Two reasons. First, the real model is trained on a 143MB CSV that can't live
in the repo. Second, tests should be independent and deterministic — if the
production model changes, tests using it would need to change too, coupling test
correctness to model quality. The conftest trains a 10-estimator XGBoost on 500
synthetic rows in about 100ms. It validates API contracts and software
correctness, not model performance.

---

### Advanced Technical Level

**Q: Walk me through how SHAP contributions are computed for a single
prediction.**
A: The model's XGBoost booster is extracted with `model.get_booster()`. A
`DMatrix` is created from the scaled 1×14 feature array. `booster.predict(dmatrix,
pred_contribs=True)` triggers TreeSHAP, which traverses each tree and computes
the contribution of each feature to each leaf assignment. The output is a 1×15
array — 14 feature contributions plus a bias term (the 15th element, the model's
expected output). The bias is excluded, and the remaining 14 are sorted by
absolute value, returning top-5.

**Q: What would break if you applied StandardScaler to the full dataset before
splitting?**
A: Two things. First, the test set's statistics (mean and std) would leak into
the scaler, meaning the model has indirectly seen properties of the test set
during training. Second, at inference time the scaler would use population
statistics that were computed with test examples included — the production scaler
would be slightly different from training, introducing a subtle distribution
shift.

**Q: Explain the Wilson confidence interval and why it's better here.**
A: The standard normal approximation `p ± z√(p(1-p)/n)` can produce a negative
lower bound at p=0.0017%. The Wilson interval inverts the acceptance region of a
z-test for each possible true value. It always stays in [0,1], has better
coverage at extreme proportions, and is what statistics textbooks recommend for
proportions below 5%.

**Q: How does `@lru_cache` on `get_bundle()` work under concurrent requests?**
A: Once the cache is populated (on the first request), subsequent calls return
the cached object with no locking needed — it's a pure read. In an async FastAPI
context with multiple uvicorn workers, each worker is a separate process and has
its own cache. Each worker loads the model once on first request. This is
acceptable — models are read-only at inference time.

---

### System Design Level

**Q: How would you scale this to 10,000 requests per second?**
A: Four changes. First, deploy multiple uvicorn workers behind a load balancer —
the API is stateless. Second, replace `@lru_cache` with a process-level singleton
loaded at startup, eliminating the first-request latency spike. Third, add a
Redis cache for repeated identical transactions (fraud rings often probe with
identical parameters). Fourth, move SHAP to an async explanation service —
compute it only on flagged transactions, not every request. The /batch-predict
endpoint already skips SHAP (`explain=False`) as a design decision.

**Q: What would you change to make this production-ready at a real bank?**
A: Six things. Replace population-level `amount_zscore` with account-level
normalisation. Add model drift detection — monitor fraud probability distribution
over time. Replace the flat-file artifact store with a proper feature store.
Add rate limiting and authentication on the API. Implement A/B testing so
threshold changes can be validated before full rollout. Move MLflow to a
PostgreSQL-backed server with S3 artifact storage.

---

### Limitation Questions

**Q: What's the biggest limitation of your current model?**
A: The `amount_zscore` is a population-level statistic. Real fraud detection
uses account-level features: "is this transaction 5x higher than your typical
purchase?" requires a rolling window per cardholder. The dataset doesn't include
cardholder IDs, so this wasn't possible, but it's the single biggest uplift
available.

**Q: What does a silhouette score of 0.105 tell you about your segments?**
A: The clusters overlap significantly — segments are not cleanly separable in
the 12-dimensional clustering space. The business value is still real (Segment 2
holds 67.9% of fraud in 32.5% of volume), but segment membership is a soft
directional signal, not a hard boundary for rules.

---

## 9. "Why Did You Build This?" Answers

### Recruiter-Friendly Version
"I wanted to build something that demonstrates the full stack of a real data
product — not just a notebook, but a system with an API, a dashboard, a
database, tests, and Docker. Fraud detection is the perfect domain because the
business problem is clear, the technical challenges are real, and the output is
something any interviewer understands immediately."

### Technical Interviewer Version
"I was specifically interested in the production gap — the difference between
training a model and deploying one responsibly. This project forced me to think
about data leakage prevention, the right evaluation metrics for imbalanced
classes, decision threshold as a business parameter rather than a statistical
default, and how to make model decisions auditable through SHAP. The statistical
testing layer was deliberate: I wanted every pattern in the executive insights
memo to be defensible."

### Startup Founder Version
"The fraud analytics space is full of black boxes. Banks have fraud scores but
can't tell customers why their card was declined. I wanted to prove the
architecture for a transparent system — one where every decision comes with an
explanation, every threshold is expressed in euros not percentages, and the
whole thing is deployable on a laptop or a cloud VM."

### Academic Evaluator Version
"The project was motivated by three methodological questions: How do you
correctly apply oversampling in a cross-validation context without introducing
leakage? How do you translate an ML performance metric into a business cost
function? And which non-parametric statistical tests are valid for the specific
data distributions in payment fraud? Each question has a concrete implementation
answer in the codebase, documented with its rationale."

---

## 10. "What Would You Improve?" Section

> Presenting these shows engineering maturity. Frame limitations as next steps,
> not failures.

### Current Limitations

**Account-level features are missing.** All engineered features are global
population statistics. Real fraud detection uses account-level features: "is
this transaction 5× higher than your typical purchase?" requires a rolling
window per cardholder. The dataset has no cardholder IDs, so this wasn't
possible here, but it's the single largest uplift available.

**Static amount_zscore.** The zscore is computed once on the training set and
hardcoded into the serving contract. In production, this would drift as spending
patterns change. This needs a feature store with a rolling compute window.

**No model drift detection.** The system has no alerting when the incoming fraud
probability distribution shifts. This is required in production — concept drift
in fraud is fast and adversarial.

### Production-Readiness Gaps

**No authentication on the API.** Any service can call `/predict`. In production:
API key auth at minimum, mTLS for internal services.

**No rate limiting.** The `/batch-predict` endpoint accepts up to 1,000
transactions per call with no rate limiting. A compromised key could cause
denial-of-service via compute exhaustion.

**MLflow model registry in local file store.** The production model is registered
in a local `mlruns/` directory. In production, this must be a database-backed
server with promotion gates (Staging → Production).

**No data validation on ingest.** `src/ingest.py` loads CSV chunks directly to
PostgreSQL. A schema validation step (Great Expectations or Pandera) should gate
data quality before it reaches the feature store.

### Scalability Concerns

**SHAP on every request.** TreeSHAP adds ~2ms per request. At 10K RPS, compute
SHAP only on flagged transactions or defer it to an async explanation service.

**Single-model serving.** The `@lru_cache` pattern works for a single model. A
multi-model system (challenger/champion, A/B variants) needs a proper model
registry with versioned loading.

**Streamlit is single-threaded.** Fine for internal analytics teams. Doesn't
scale to external users. For a customer-facing interface, replace it with a
React/Vue frontend consuming the FastAPI backend.

### Future Enhancements

1. Real-time feature engineering via Kafka + Faust for account-level velocity
2. Periodic retraining triggered when performance metrics drop below threshold
3. Graph-based features — merchant-cardholder interaction graphs surface fraud
   rings invisible to transaction-level models
4. Probability calibration — post-SMOTE probabilities don't reflect the true
   0.17% base rate; Platt scaling or isotonic calibration would fix this

---

## 11. Resume Bullet Points

### 3 Concise Bullets

- Built end-to-end fraud detection system (XGBoost + SMOTE) achieving 78.9%
  recall and 0.967 AUC-ROC on 284K imbalanced transactions (0.17% fraud rate)
- Engineered cost-based decision threshold optimizer reducing expected fraud
  costs by 38.5% (€4,489 savings vs. default 0.5 threshold)
- Delivered production ML stack: FastAPI scoring service with real-time SHAP
  explanations, Streamlit analytics dashboard, PostgreSQL BI layer, Docker
  Compose deployment, and 41 automated tests

### 5 Impact-Focused Bullets

- Detected credit card fraud with 78.9% recall at 0.967 AUC-ROC, rendering
  the 99.8%-accurate all-negative baseline irrelevant through correct metric
  selection (average precision)
- Reduced expected business cost by 38.5% through cost-function-driven
  threshold optimisation, translating ML parameters into €4,489 savings on
  test-window data
- Implemented exact TreeSHAP explanations on every API prediction, making 100%
  of model decisions auditable — critical for regulatory compliance in financial
  services
- Validated all discovered patterns with formal statistical tests: chi-square
  confirms 3.25× elevated night-time risk (p=10⁻³²), Mann-Whitney U confirms
  fraud skews smaller (median €9.82 vs €22.00, p=10⁻⁵)
- Built test suite covering full API contract, inference pipeline, and analytics
  modules with zero dataset dependency — CI runs in under 2 seconds with 41
  passing tests

### 10 ATS-Friendly Bullets

- Developed end-to-end machine learning pipeline for credit card fraud detection
  using XGBoost, SMOTE, and scikit-learn on 284,807 transaction records
- Achieved 78.9% fraud recall and 0.967 AUC-ROC with 0.746 average precision
  on held-out test set using stratified split and imbalanced-learn
- Implemented cost-based decision threshold optimisation sweeping 99 threshold
  values to minimise real business cost in EUR (missed fraud + false decline
  friction)
- Built RESTful API using FastAPI and Pydantic v2 with five endpoints including
  real-time SHAP explanations and batch scoring of up to 1,000 transactions
- Integrated XGBoost native TreeSHAP via pred_contribs=True for per-prediction
  feature attribution in log-odds space on every API response
- Conducted four-way model comparison (SMOTE, class weighting, undersampling,
  Isolation Forest) using MLflow experiment tracking with logged params and
  metrics
- Applied statistical hypothesis testing: chi-square, Mann-Whitney U, and
  Wilson score confidence intervals to validate fraud patterns quantitatively
- Designed K-means behavioural segmentation (4 clusters) identifying segment
  concentrating 67.9% of fraud in 32.5% of transaction volume
- Authored 41 automated tests covering API contracts, model inference, and
  threshold optimiser using pytest and FastAPI TestClient with no dataset
  dependency
- Containerised full stack including PostgreSQL 16, MLflow, FastAPI, and
  Streamlit using Docker Compose with health checks and service dependencies

---

## 12. LinkedIn Project Description

### Short Version (100 words)

**FinGuard — Real-Time Fraud Detection System**

Built a production-grade fraud detection platform on 284,807 real credit card
transactions (0.17% fraud rate). XGBoost + SMOTE achieves 78.9% fraud recall
and 0.967 AUC-ROC. A cost-based threshold optimiser reduced expected fraud costs
by 38.5% versus the standard 0.5 default. Every API prediction ships with a
real-time SHAP explanation. Stack: Python · XGBoost · FastAPI · Streamlit ·
PostgreSQL · MLflow · Docker.

---

### Medium Version (200 words)

**FinGuard — Production Fraud Detection with Explainable AI**

Most fraud models optimise accuracy — which is meaningless when 99.8% of
transactions are legitimate. FinGuard reframes the problem around money: how
many euros does the model save?

Built on 284,807 real transactions, the system uses XGBoost with SMOTE
oversampling applied correctly — strictly after the train/test split to prevent
data leakage — achieving 78.9% recall and 0.967 AUC-ROC. A cost-based threshold
optimiser sweeps 99 decision points and minimises expected cost using real
transaction amounts, resulting in a 38.5% cost reduction versus the default.

Every prediction from the FastAPI scoring service includes a real-time SHAP
explanation via XGBoost's native TreeSHAP, making every decision auditable. The
analytics layer includes formal statistical validation: chi-square confirms
3.25× elevated night-time fraud risk (p≈10⁻³²), and K-means segmentation
identifies a cluster holding 67.9% of all fraud in 32.5% of volume.

Full stack: FastAPI · Pydantic v2 · Streamlit · PostgreSQL · MLflow · Docker
Compose · 41 automated tests.

---

### Portfolio Version (Full Technical Depth)

**FinGuard: End-to-End ML Fraud Detection System**

A production-grade, fully explainable fraud detection system — from raw CSV to
a containerised API with real-time SHAP explanations.

**The challenge:** 284,807 transactions, 492 fraud cases (0.17%). Accuracy is
useless. Average precision is the metric. SMOTE must be applied after the split.
The 0.5 decision threshold loses money.

**What I built:**
- ML pipeline with leakage prevention: stratified split → fit-on-train
  StandardScaler → SMOTE on training only → XGBoost with aucpr eval metric
- Cost-based threshold optimiser: 99-point sweep minimising
  `missed_fraud_EUR + FP_cost × false_declines` — optimal threshold 0.86 saves
  €4,489 vs default
- FastAPI service (5 endpoints) with `@lru_cache` model loading and XGBoost
  TreeSHAP via `pred_contribs=True` on every prediction
- Four-way imbalance strategy bake-off in MLflow (SMOTE, class weighting,
  undersampling, Isolation Forest) — experiment-backed decision
- Statistical validation: chi-square (3.25× night risk, p=10⁻³²),
  Mann-Whitney U (fraud median €9.82 vs €22, p=10⁻⁵), Wilson CI
- K-means segmentation: 32.5% of volume containing 67.9% of fraud
- 41 automated tests with zero dataset dependency

**Results:** 78.9% recall · 0.967 AUC-ROC · 0.746 average precision ·
38.5% cost reduction

**Tech:** Python · XGBoost · FastAPI · Pydantic v2 · Streamlit · PostgreSQL ·
MLflow · Docker Compose · pytest

---

## 13. Recruiter Evaluation Perspective

### What a Technical Recruiter Will Notice

**Positive signals:**
- Full-stack ownership: data pipeline, ML, API, dashboard, database,
  containerisation, testing — demonstrates end-to-end capability
- Business framing: costs in euros rather than abstract metrics — shows
  commercial awareness
- Real dataset with a well-known benchmark (Kaggle Credit Card Fraud)
- Deployed and demonstrable

**Potential concern:** "Did they understand the code or just run it?"

Prep: Use your STAR stories from Section 7. Explain *why* SMOTE goes after the
split, *why* Mann-Whitney over t-test, *why* Wilson CI. These show depth.

---

### What a Software Engineering Interviewer Will Notice

**Positive signals:**
- SMOTE placement shows awareness of a common and subtle data leakage pattern
- `@lru_cache(maxsize=1)` is the right pattern for a model singleton
- Wilson interval choice shows statistical literacy beyond defaults
- Tests are independent of the real dataset — professionally structured

**What they'll probe:**
- "Walk me through what happens when `/predict` gets called" — use Section 3.2
- "How would you scale to 10K RPS?" — use Section 8 system design answer
- "Why not sklearn predict_proba for SHAP?" — XGBoost native is exact;
  SHAP library is an approximation wrapper

---

### What a Hiring Manager Will Notice

**Positive signals:**
- The project demonstrates judgment: SMOTE after split, cost-based threshold,
  correct statistical tests — these are decisions, not just implementations
- The executive insights format shows understanding of who consumes ML outputs
- 41 tests with a clean CI story says "this person ships maintainable code"

**How to strengthen the presentation:**
- Lead with the business outcome (€4,489 saved) before the technical metrics
- Mention the 4-way bake-off early — it shows you don't assume, you experiment
- Have the live demo running before the interview starts; showing real-time SHAP
  explanations is memorable

---

## 14. Demonstration Script (5–7 Minutes)

> Start both servers before the interview begins.
> FastAPI: `uvicorn api.main:app --port 8000`
> Dashboard: `streamlit run dashboard/app.py`

**[0:00 – 0:45] Opening**

"What I'm about to show you is FinGuard — a fraud detection system I built
end-to-end. The problem: in 284,000 real credit card transactions, only 492 are
fraud — 0.17%. A model that approves everything is 99.8% accurate and catches
nothing. This project is about building something that actually catches fraud,
explains every decision, and expresses everything in business terms — euros,
not percentages."

**[0:45 – 1:30] Home Page**

"The home page shows the live metrics from the training run. 78.9% recall —
nearly 4 in 5 frauds caught. Average precision of 0.746 — the honest metric
at a 0.17% base rate. And the cost-optimised threshold has already saved
€4,489 compared to the statistical default of 0.5. I'll show you how that
works in a moment."

**[1:30 – 2:30] Executive Insights**

"The insights page is the analyst's memo. Every finding here is backed by a
statistical test. [Point to night-time finding] Night-time transactions have a
3.25× higher fraud rate — confirmed by chi-square at p=10⁻³², not eyeballed.
[Point to amount finding] Fraud amounts skew smaller than legitimate ones —
median €9.82 vs €22. That comes from a Mann-Whitney U test, not a t-test,
because transaction amounts are not normally distributed. The statistical choice
is documented in the code."

**[2:30 – 3:30] Model Performance — Cost Threshold Tab**

"Here's the business case for the threshold. [Show cost curve] At threshold 0.5,
the system incurs €11,670 on the test window. At threshold 0.86 — the minimum
— it incurs €7,181. That's 38.5% saved. The cost model uses real transaction
amounts for missed fraud and €30 per false decline. Change FP_COST in the
environment and the optimizer recalibrates automatically."

**[3:30 – 4:30] Strategy Bake-off Tab**

"I ran four imbalance strategies on identical data, logged in MLflow. [Show
table] Class weighting beat SMOTE on average precision — 0.759 vs 0.746 — while
more than doubling precision. The production model uses SMOTE because recall is
slightly higher, but the decision is backed by evidence. The unsupervised
Isolation Forest gap quantifies the value of having fraud labels."

**[4:30 – 5:30] Live Prediction**

"[Select 'Fraud-pattern probe' preset] This uses PCA values that resemble known
fraud patterns — small amount, 3am, anomalous behavioural signals. [Hit Score]
99.95% fraud probability. CRITICAL. [Point to SHAP chart] The chart shows why:
V10, V4, and V3 are the top contributors. These are XGBoost's native TreeSHAP
values — exact, not approximate. Every prediction from this API returns these."

**[5:30 – 6:00] Close**

"The full stack is containerised — one `docker-compose up` starts PostgreSQL,
MLflow, the API, and the dashboard. 41 automated tests, no dataset dependency,
CI runs in under 2 seconds. The thing I'd improve next is account-level
features — the amount z-score should be relative to your own spending history,
not the population. That's the single biggest available uplift."

---

## 15. Cheat Sheet

```
FINGUARD — ONE-PAGE CHEAT SHEET
────────────────────────────────────────────────────────────────
DATASET
  284,807 transactions · 492 fraud (0.17%) · Kaggle Credit Card Fraud

ARCHITECTURE
  CSV → preprocess → train → artifacts/
  artifacts/ → FastAPI (5 endpoints) → Streamlit (7 pages)
  PostgreSQL ← ingest (15 SQL queries) ← CSV
  Docker Compose: PostgreSQL 16 · MLflow · API · Dashboard

ML STACK
  Model        XGBoost (n_estimators=100, max_depth=6, lr=0.1,
               tree_method=hist, eval_metric=aucpr)
  Imbalance    SMOTE applied AFTER split on training data only
  Features     14: V1-V10, Amount, hour, is_night, zscore
  Validation   Pydantic v2 field_validator (len(v)==10, etc.)
  SHAP         XGBoost native pred_contribs=True (exact TreeSHAP)

KEY METRICS
  Recall               78.9%       "4 in 5 frauds caught"
  Precision            24.6%       "145× lift over random flag"
  Average Precision    0.746       headline metric at 0.17% rate
  AUC-ROC              0.967       (inflated by imbalance, caveat)
  Optimal threshold    0.86        cost model minimum
  Cost saving          €4,489 / 38.5% vs default 0.5 threshold
  Best bake-off        scale_pos_weight (AP=0.759 vs SMOTE 0.746)

STATISTICAL TESTS
  Chi-square   night risk 3.25× (χ²=141.3, p=1.38×10⁻³²)
  Mann-Whitney fraud median €9.82 vs legit €22 (p=2.69×10⁻⁵)
  Wilson CI    per-hour rates (normal CI goes negative at 0.17%)

SEGMENTS (K-means k=4, silhouette=0.105)
  Segment 2: 32.5% of volume → 67.9% of fraud (2.1× baseline)

TESTS
  41 tests · pytest · zero dataset dependency
  conftest trains 10-estimator XGBoost on 500 synthetic rows

KEY Q&A (30-second answers)
  "Why not accuracy?"      Useless at 0.17%; use PR-AUC
  "Why XGBoost?"           Exact TreeSHAP, hist fast, tabular SOTA
  "SMOTE after split?"     Prevents leakage of synthetic test neighbours
  "Why 0.86 threshold?"    Cost model: missed fraud >> FP €30 friction
  "Why Mann-Whitney?"      Amount non-normal; t-test invalid
  "Scale to 10K RPS?"      Stateless API, horizontal scale, async SHAP

WHAT YOU'D IMPROVE
  1. Account-level features (biggest single uplift)
  2. Model drift detection/alerting
  3. Auth + rate limiting on API
  4. Probability calibration post-SMOTE
  5. PostgreSQL-backed MLflow with S3 artifacts
────────────────────────────────────────────────────────────────
```

---

## 16. AI-Assisted Development Disclosure Strategy

> Read this before any interview. Honesty about tooling, combined with
> demonstrated understanding, is always the right strategy.

### What to Say if Asked Whether AI Tools Were Used

**Honest, professional answer:**

"Yes, I used Claude Code as a development assistant during parts of this project
— similar to how engineers use GitHub Copilot or Stack Overflow. It helped with
boilerplate, CSS styling, and debugging specific errors like Pydantic namespace
warnings and Streamlit version compatibility issues. But every architectural
decision — SMOTE placement, the cost-based threshold, the statistical test
selection, the SHAP integration — I made and can explain from first principles.
If you want, walk me through any part of the code and I'll explain the rationale."

**What NOT to say:**
- "No, I wrote everything myself" — if untrue, this is a serious integrity risk
- "The AI built the whole thing" — undersells your actual judgment and learning
- "I just prompted it and copied the output" — suggests no understanding

### How to Demonstrate Genuine Understanding

The questions that prove understanding are always *why*, not *what*:

| Question | The answer that proves depth |
|----------|------------------------------|
| Why is SMOTE applied after the split? | Leakage: synthetic samples interpolate between test-set neighbours |
| Why Wilson CI not normal approximation? | Normal CI goes negative at p=0.17% |
| Why `eval_metric='aucpr'` in XGBoost? | Aligns training signal with correct evaluation criterion |
| Why `@lru_cache(maxsize=1)` not a global? | Works with FastAPI's DI, clearable in tests |
| Why `pred_contribs=True` not SHAP library? | No extra dependency; XGBoost native TreeSHAP is exact |
| Why Mann-Whitney not t-test? | Transaction amounts are right-skewed, t-test normality fails |

Prepare to answer all of these fluently. They are all answered in Sections 5 and 8.

### How to Emphasise Engineering Judgment

The decisions that are genuinely yours to own:

1. **The cost function design** — `missed_fraud + FP_cost × false_positives` is
   a business judgment about what to model, not a default
2. **Statistical test selection** — Mann-Whitney over t-test, Wilson over normal
   CI — deliberate choices with documented rationales in the code
3. **The 4-way bake-off** — designing the experiment is a judgment call; letting
   the results change your conclusion (class weighting beats SMOTE) is the
   intellectual honesty that impresses interviewers
4. **SHAP in every API response** — a product decision: every declined
   transaction must be explainable
5. **The artifact-first architecture** — decoupling training and serving so
   the API never touches training code is a system design choice

### What Statements to Avoid

| Don't say | Say instead |
|-----------|-------------|
| "The AI wrote this function" | "This function implements X, which I needed because Y" |
| "I'm not sure why it works this way" | "I'd need to look at the code — let me walk through it" |
| "I just followed what it suggested" | "I evaluated a few approaches and chose this because [reason]" |
| "I couldn't have built this without AI" | "AI tooling accelerated development; the design decisions are mine" |

### The Core Principle

Modern software engineering uses tools. AI assistants are tools. What
interviewers are evaluating is whether you understand what you built, can defend
the decisions, debug problems, and extend the system. This document exists
precisely so you can do all four. The actual codebase contains dozens of
defensible decisions documented in the sections above. Own them.

---

*Generated from full codebase analysis — every metric, threshold, and code
reference is verified against the actual implementation in this repository.*
