"""SQL-based fraud analysis on SQLite via SQLAlchemy.

Loads the transaction CSV into output/fraud.db and runs four analytical queries:
    1. Fraud rate by merchant category
    2. High-risk time windows (day of week x 4-hour block)
    3. Amount anomaly detection (amount far above the customer's 30-day average)
    4. Repeat fraud patterns (recurring fraud profiles and back-to-back fraud bursts)

Usage:
    python src/sql_analysis.py
"""

import pandas as pd
from sqlalchemy import create_engine, text

from common import DATA_PATH, OUTPUT_DIR

DB_PATH = OUTPUT_DIR / "fraud.db"

QUERIES = {
    "1. Fraud rate by merchant category": """
        SELECT merchant_category,
               COUNT(*)                                   AS txns,
               SUM(is_fraud)                              AS frauds,
               ROUND(100.0 * AVG(is_fraud), 2)            AS fraud_rate_pct,
               ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2) AS fraud_amount,
               ROUND(AVG(CASE WHEN is_fraud = 1 THEN amount END), 2)        AS avg_fraud_amount
        FROM transactions
        GROUP BY merchant_category
        ORDER BY fraud_rate_pct DESC
    """,

    "2. High-risk time windows (top 10, min 500 txns)": """
        WITH windows AS (
            SELECT CASE day_of_week
                       WHEN 0 THEN 'Mon' WHEN 1 THEN 'Tue' WHEN 2 THEN 'Wed'
                       WHEN 3 THEN 'Thu' WHEN 4 THEN 'Fri' WHEN 5 THEN 'Sat' ELSE 'Sun'
                   END AS day,
                   printf('%02d:00-%02d:59', (hour_of_day / 4) * 4, (hour_of_day / 4) * 4 + 3) AS time_block,
                   is_fraud
            FROM transactions
        ),
        overall AS (SELECT AVG(is_fraud) AS rate FROM transactions)
        SELECT day, time_block,
               COUNT(*)                                  AS txns,
               SUM(is_fraud)                             AS frauds,
               ROUND(100.0 * AVG(is_fraud), 2)           AS fraud_rate_pct,
               ROUND(AVG(is_fraud) / (SELECT rate FROM overall), 2) AS risk_vs_overall
        FROM windows
        GROUP BY day, time_block
        HAVING COUNT(*) >= 500
        ORDER BY fraud_rate_pct DESC
        LIMIT 10
    """,

    "3. Amount anomaly detection (amount / 30-day avg)": """
        WITH ratios AS (
            SELECT is_fraud,
                   CASE
                       WHEN amount / avg_amount_last_30d < 1  THEN '1: below avg'
                       WHEN amount / avg_amount_last_30d < 2  THEN '2: 1-2x avg'
                       WHEN amount / avg_amount_last_30d < 3  THEN '3: 2-3x avg'
                       WHEN amount / avg_amount_last_30d < 5  THEN '4: 3-5x avg'
                       WHEN amount / avg_amount_last_30d < 10 THEN '5: 5-10x avg'
                       ELSE                                        '6: 10x+ avg'
                   END AS ratio_band
            FROM transactions
        )
        SELECT ratio_band,
               COUNT(*)                        AS txns,
               SUM(is_fraud)                   AS frauds,
               ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct,
               ROUND(100.0 * SUM(is_fraud) / (SELECT SUM(is_fraud) FROM transactions), 2) AS pct_of_all_fraud
        FROM ratios
        GROUP BY ratio_band
        ORDER BY ratio_band
    """,

    "3b. Top 10 anomalous transactions (z-score of amount within category)": """
        WITH stats AS (
            SELECT merchant_category,
                   AVG(amount) AS mu,
                   SQRT(AVG(amount * amount) - AVG(amount) * AVG(amount)) AS sigma
            FROM transactions
            GROUP BY merchant_category
        )
        SELECT t.transaction_id, t.merchant_category, t.amount, t.avg_amount_last_30d,
               ROUND((t.amount - s.mu) / s.sigma, 2) AS z_score,
               t.is_foreign, t.is_fraud
        FROM transactions t
        JOIN stats s USING (merchant_category)
        ORDER BY z_score DESC
        LIMIT 10
    """,

    "4. Repeat fraud patterns: recurring fraud profiles (min 15 frauds)": """
        SELECT merchant_category,
               CASE WHEN is_foreign = 1 THEN 'foreign' ELSE 'domestic' END AS origin,
               CASE WHEN hour_of_day < 6 THEN 'night' ELSE 'day' END     AS period,
               CASE WHEN previous_transaction_gap_hours < 1 THEN '<1h' ELSE '>=1h' END AS gap,
               COUNT(*)                        AS txns,
               SUM(is_fraud)                   AS frauds,
               ROUND(100.0 * AVG(is_fraud), 2) AS fraud_rate_pct
        FROM transactions
        GROUP BY 1, 2, 3, 4
        HAVING SUM(is_fraud) >= 15
        ORDER BY fraud_rate_pct DESC
        LIMIT 10
    """,

    "4b. Repeat fraud patterns: back-to-back fraud bursts in the transaction stream": """
        WITH seq AS (
            SELECT transaction_id, is_fraud, merchant_category,
                   ROW_NUMBER() OVER (ORDER BY transaction_id) AS rn
            FROM transactions
        ),
        frauds AS (
            SELECT transaction_id, merchant_category, rn,
                   rn - ROW_NUMBER() OVER (ORDER BY rn) AS grp
            FROM seq
            WHERE is_fraud = 1
        )
        SELECT MIN(transaction_id)                AS first_txn,
               MAX(transaction_id)                AS last_txn,
               COUNT(*)                           AS burst_length,
               GROUP_CONCAT(merchant_category, ', ') AS categories
        FROM frauds
        GROUP BY grp
        HAVING COUNT(*) >= 2
        ORDER BY burst_length DESC, first_txn
        LIMIT 10
    """,
}


def load_into_sqlite(engine) -> int:
    df = pd.read_csv(DATA_PATH)
    df.to_sql("transactions", engine, if_exists="replace", index=False, chunksize=10_000)
    with engine.begin() as conn:
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_category ON transactions (merchant_category)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_fraud ON transactions (is_fraud)"))
    return len(df)


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found. Run `python data/generate_data.py` first.")
    OUTPUT_DIR.mkdir(exist_ok=True)
    engine = create_engine(f"sqlite:///{DB_PATH}")
    n = load_into_sqlite(engine)
    print(f"Loaded {n:,} rows into {DB_PATH} (table: transactions)")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    with engine.connect() as conn:
        for title, sql in QUERIES.items():
            result = pd.read_sql(text(sql), conn)
            print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
            print(result.to_string(index=False))

        # Summary line for the burst query: how many bursts exist overall, vs. what chance would give.
        bursts = conn.execute(text("""
            SELECT COUNT(*) FROM (
                SELECT transaction_id, is_fraud,
                       LAG(is_fraud) OVER (ORDER BY transaction_id) AS prev_fraud
                FROM transactions
            ) WHERE is_fraud = 1 AND prev_fraud = 1
        """)).scalar()
        rate = conn.execute(text("SELECT AVG(is_fraud) FROM transactions")).scalar()
    expected = rate * rate * (n - 1)
    print(f"\nConsecutive fraud pairs: {bursts} (about {expected:.0f} expected by chance)")


if __name__ == "__main__":
    main()
