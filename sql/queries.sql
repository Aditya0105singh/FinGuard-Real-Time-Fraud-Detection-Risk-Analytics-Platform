-- ════════════════════════════════════════════════════════════════════
-- FinGuard — Business Analytics Queries
-- Table: transactions
--   Columns: id, time_offset, v1..v28, amount, class,
--            fraud_label, risk_tier, transaction_hour, day_of_week,
--            transaction_timestamp
--
-- NOTE: the Kaggle dataset spans ~48 hours of activity. ingest.py
-- derives transaction_timestamp = '2024-01-01' + time_offset seconds
-- so that date-window queries (MoM, YTD, rolling) are demonstrable.
-- ════════════════════════════════════════════════════════════════════


-- ────────────────────────────────────────────────────────────────────
-- 1. Overall fraud rate and total transactions
--    Headline KPI: how big is the fraud problem?
-- ────────────────────────────────────────────────────────────────────
SELECT
    COUNT(*)                                          AS total_transactions,
    SUM(fraud_label)                                  AS fraud_count,
    ROUND(100.0 * SUM(fraud_label) / COUNT(*), 4)     AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN fraud_label = 1 THEN amount ELSE 0 END)::numeric, 2)
                                                      AS total_fraud_amount
FROM transactions;


-- ────────────────────────────────────────────────────────────────────
-- 2. Fraud rate by transaction hour (0–23)
--    Identifies which hours of day carry elevated fraud risk.
-- ────────────────────────────────────────────────────────────────────
SELECT
    transaction_hour,
    COUNT(*)                                          AS txn_count,
    SUM(fraud_label)                                  AS fraud_count,
    ROUND(100.0 * SUM(fraud_label) / COUNT(*), 4)     AS fraud_rate_pct
FROM transactions
GROUP BY transaction_hour
ORDER BY transaction_hour;


-- ────────────────────────────────────────────────────────────────────
-- 3. Fraud rate by day of week (0 = first day in the capture window)
-- ────────────────────────────────────────────────────────────────────
SELECT
    day_of_week,
    COUNT(*)                                          AS txn_count,
    SUM(fraud_label)                                  AS fraud_count,
    ROUND(100.0 * SUM(fraud_label) / COUNT(*), 4)     AS fraud_rate_pct
FROM transactions
GROUP BY day_of_week
ORDER BY day_of_week;


-- ────────────────────────────────────────────────────────────────────
-- 4. Top 10 highest-amount fraud transactions
--    The biggest single losses — candidates for case review.
-- ────────────────────────────────────────────────────────────────────
SELECT
    id,
    transaction_timestamp,
    transaction_hour,
    amount,
    risk_tier
FROM transactions
WHERE fraud_label = 1
ORDER BY amount DESC
LIMIT 10;


-- ────────────────────────────────────────────────────────────────────
-- 5. Average transaction amount: fraud vs legitimate
--    Do fraudsters transact at different ticket sizes?
-- ────────────────────────────────────────────────────────────────────
SELECT
    CASE WHEN fraud_label = 1 THEN 'FRAUD' ELSE 'LEGITIMATE' END AS txn_type,
    COUNT(*)                       AS txn_count,
    ROUND(AVG(amount)::numeric, 2) AS avg_amount,
    ROUND(MIN(amount)::numeric, 2) AS min_amount,
    ROUND(MAX(amount)::numeric, 2) AS max_amount,
    ROUND(STDDEV(amount)::numeric, 2) AS stddev_amount
FROM transactions
GROUP BY fraud_label;


-- ────────────────────────────────────────────────────────────────────
-- 6. Running fraud count over time (window function — ROW_NUMBER)
--    Sequence number of each fraud event as it occurred.
-- ────────────────────────────────────────────────────────────────────
SELECT
    id,
    transaction_timestamp,
    amount,
    ROW_NUMBER() OVER (ORDER BY transaction_timestamp) AS fraud_sequence_no
FROM transactions
WHERE fraud_label = 1
ORDER BY transaction_timestamp;


-- ────────────────────────────────────────────────────────────────────
-- 7. Transaction amount percentiles (NTILE into 4 buckets)
--    Quartile segmentation of spend with fraud rate per quartile.
-- ────────────────────────────────────────────────────────────────────
WITH bucketed AS (
    SELECT
        amount,
        fraud_label,
        NTILE(4) OVER (ORDER BY amount) AS amount_quartile
    FROM transactions
)
SELECT
    amount_quartile,
    COUNT(*)                                      AS txn_count,
    ROUND(MIN(amount)::numeric, 2)                AS bucket_min,
    ROUND(MAX(amount)::numeric, 2)                AS bucket_max,
    SUM(fraud_label)                              AS fraud_count,
    ROUND(100.0 * SUM(fraud_label) / COUNT(*), 4) AS fraud_rate_pct
FROM bucketed
GROUP BY amount_quartile
ORDER BY amount_quartile;


-- ────────────────────────────────────────────────────────────────────
-- 8. Month-over-month fraud trend (CTE)
--    With the synthetic timestamp the data covers one month; the
--    pattern generalises directly to production date ranges.
-- ────────────────────────────────────────────────────────────────────
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', transaction_timestamp) AS month,
        COUNT(*)                                   AS txn_count,
        SUM(fraud_label)                           AS fraud_count,
        SUM(CASE WHEN fraud_label = 1 THEN amount ELSE 0 END) AS fraud_amount
    FROM transactions
    GROUP BY DATE_TRUNC('month', transaction_timestamp)
)
SELECT
    month,
    txn_count,
    fraud_count,
    ROUND(fraud_amount::numeric, 2)               AS fraud_amount,
    ROUND(100.0 * fraud_count / txn_count, 4)     AS fraud_rate_pct
FROM monthly
ORDER BY month;


-- ────────────────────────────────────────────────────────────────────
-- 9. High-risk amount ranges (CASE WHEN buckets)
--    Which spend bands concentrate fraud?
-- ────────────────────────────────────────────────────────────────────
SELECT
    CASE
        WHEN amount = 0              THEN '0 — zero-amount probe'
        WHEN amount <= 10            THEN '0.01 – 10'
        WHEN amount <= 100           THEN '10 – 100'
        WHEN amount <= 500           THEN '100 – 500'
        WHEN amount <= 1000          THEN '500 – 1,000'
        ELSE                              '1,000+'
    END                                           AS amount_band,
    COUNT(*)                                      AS txn_count,
    SUM(fraud_label)                              AS fraud_count,
    ROUND(100.0 * SUM(fraud_label) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY 1
ORDER BY fraud_rate_pct DESC;


-- ────────────────────────────────────────────────────────────────────
-- 10. Fraud rate rolling average (LAG function)
--     Hour-by-hour fraud rate with previous-period comparison and
--     a trailing window average over the prior 7 periods.
-- ────────────────────────────────────────────────────────────────────
WITH hourly AS (
    SELECT
        DATE_TRUNC('hour', transaction_timestamp)     AS hour_bucket,
        COUNT(*)                                      AS txn_count,
        SUM(fraud_label)                              AS fraud_count,
        100.0 * SUM(fraud_label) / COUNT(*)           AS fraud_rate_pct
    FROM transactions
    GROUP BY 1
)
SELECT
    hour_bucket,
    ROUND(fraud_rate_pct, 4)                          AS fraud_rate_pct,
    ROUND(LAG(fraud_rate_pct) OVER (ORDER BY hour_bucket), 4)
                                                      AS prev_period_rate,
    ROUND(AVG(fraud_rate_pct) OVER (
        ORDER BY hour_bucket ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
    ), 4)                                             AS rolling_7_period_avg
FROM hourly
ORDER BY hour_bucket;


-- ────────────────────────────────────────────────────────────────────
-- 11. Cumulative fraud amount over time (SUM OVER)
--     Running total of money lost to fraud — the "bleed curve".
-- ────────────────────────────────────────────────────────────────────
SELECT
    id,
    transaction_timestamp,
    amount,
    ROUND(SUM(amount) OVER (ORDER BY transaction_timestamp)::numeric, 2)
        AS cumulative_fraud_amount
FROM transactions
WHERE fraud_label = 1
ORDER BY transaction_timestamp;


-- ────────────────────────────────────────────────────────────────────
-- 12. Rank transactions by amount within each hour (RANK)
--     Top-3 largest transactions in every hour of day.
-- ────────────────────────────────────────────────────────────────────
WITH ranked AS (
    SELECT
        id,
        transaction_hour,
        amount,
        fraud_label,
        RANK() OVER (PARTITION BY transaction_hour ORDER BY amount DESC)
            AS amount_rank_in_hour
    FROM transactions
)
SELECT *
FROM ranked
WHERE amount_rank_in_hour <= 3
ORDER BY transaction_hour, amount_rank_in_hour;


-- ────────────────────────────────────────────────────────────────────
-- 13. Transactions where amount > 2x average (subquery)
--     The dataset is anonymised (no customer id), so the population
--     average stands in for the per-user average used in production.
-- ────────────────────────────────────────────────────────────────────
SELECT
    id,
    transaction_timestamp,
    amount,
    fraud_label,
    risk_tier
FROM transactions
WHERE amount > 2 * (SELECT AVG(amount) FROM transactions)
ORDER BY amount DESC
LIMIT 50;


-- ────────────────────────────────────────────────────────────────────
-- 14. Peak fraud hours identification (HAVING clause)
--     Hours whose fraud rate exceeds 2x the overall fraud rate.
-- ────────────────────────────────────────────────────────────────────
SELECT
    transaction_hour,
    COUNT(*)                                      AS txn_count,
    SUM(fraud_label)                              AS fraud_count,
    ROUND(100.0 * SUM(fraud_label) / COUNT(*), 4) AS fraud_rate_pct
FROM transactions
GROUP BY transaction_hour
HAVING (100.0 * SUM(fraud_label) / COUNT(*)) >
       2 * (SELECT 100.0 * SUM(fraud_label) / COUNT(*) FROM transactions)
ORDER BY fraud_rate_pct DESC;


-- ────────────────────────────────────────────────────────────────────
-- 15. Year-to-date fraud summary with growth rate (CTE + LAG)
--     Daily fraud totals with day-over-day growth percentage —
--     the same shape a real YTD monthly report would use.
-- ────────────────────────────────────────────────────────────────────
WITH daily AS (
    SELECT
        DATE_TRUNC('day', transaction_timestamp)  AS day,
        COUNT(*)                                  AS txn_count,
        SUM(fraud_label)                          AS fraud_count,
        SUM(CASE WHEN fraud_label = 1 THEN amount ELSE 0 END) AS fraud_amount
    FROM transactions
    WHERE transaction_timestamp >= DATE_TRUNC('year', CURRENT_DATE)
       OR TRUE  -- synthetic timestamps are fixed to 2024; keep all rows
    GROUP BY 1
),
with_growth AS (
    SELECT
        day,
        txn_count,
        fraud_count,
        fraud_amount,
        LAG(fraud_amount) OVER (ORDER BY day) AS prev_day_fraud_amount
    FROM daily
)
SELECT
    day,
    txn_count,
    fraud_count,
    ROUND(fraud_amount::numeric, 2)                       AS fraud_amount,
    ROUND(SUM(fraud_amount) OVER (ORDER BY day)::numeric, 2)
                                                          AS ytd_fraud_amount,
    ROUND(
        CASE
            WHEN prev_day_fraud_amount IS NULL OR prev_day_fraud_amount = 0
                THEN NULL
            ELSE 100.0 * (fraud_amount - prev_day_fraud_amount)
                 / prev_day_fraud_amount
        END::numeric, 2
    )                                                     AS growth_rate_pct
FROM with_growth
ORDER BY day;
