"""Load creditcard.csv into PostgreSQL with derived analytics columns.

Creates table `transactions` with:
  - original columns (time_offset, v1..v28, amount, class)
  - fraud_label        : copy of class (explicit business name)
  - risk_tier          : LOW / MEDIUM / HIGH / CRITICAL by amount band
  - transaction_hour   : hour of day derived from the time offset
  - day_of_week        : day index derived from the time offset
  - transaction_timestamp : synthetic timestamp anchored at 2024-01-01
                            (dataset only records seconds elapsed)

Usage:
    python -m src.ingest [--csv data/creditcard.csv]
"""

import argparse
import logging
import os
import sys

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("finguard.ingest")

TABLE_NAME = "transactions"
ANCHOR_TIMESTAMP = pd.Timestamp("2024-01-01 00:00:00")
CHUNK_SIZE = 50_000


def get_engine():
    """Build a SQLAlchemy engine from the DATABASE_URL env variable."""
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError(
            "DATABASE_URL is not set. Copy .env.example to .env and configure it."
        )
    return create_engine(database_url)


def assign_risk_tier(amount: float) -> str:
    """Business rule: tier transactions by amount exposure."""
    if amount <= 50:
        return "LOW"
    if amount <= 250:
        return "MEDIUM"
    if amount <= 1000:
        return "HIGH"
    return "CRITICAL"


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived analytics columns to a raw chunk."""
    df = df.rename(columns={c: c.lower() for c in df.columns})
    df = df.rename(columns={"time": "time_offset"})

    df["fraud_label"] = df["class"].astype(int)
    df["transaction_hour"] = ((df["time_offset"] // 3600) % 24).astype(int)
    df["day_of_week"] = ((df["time_offset"] // 86400) % 7).astype(int)
    df["transaction_timestamp"] = ANCHOR_TIMESTAMP + pd.to_timedelta(
        df["time_offset"], unit="s"
    )
    df["risk_tier"] = df["amount"].apply(assign_risk_tier)
    return df


def create_indexes(engine) -> None:
    """Create indexes used by the analytics queries in sql/queries.sql."""
    statements = [
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_amount ON {TABLE_NAME} (amount)",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_class ON {TABLE_NAME} (class)",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_time ON {TABLE_NAME} (time_offset)",
        f"CREATE INDEX IF NOT EXISTS idx_{TABLE_NAME}_timestamp "
        f"ON {TABLE_NAME} (transaction_timestamp)",
    ]
    with engine.begin() as conn:
        for stmt in statements:
            logger.info("Executing: %s", stmt)
            conn.execute(text(stmt))


def ingest(csv_path: str) -> int:
    """Stream the CSV into PostgreSQL in chunks. Returns rows loaded."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. See data/README.md for download instructions."
        )

    engine = get_engine()
    total_rows = 0

    logger.info("Loading %s into table '%s' ...", csv_path, TABLE_NAME)
    for i, chunk in enumerate(pd.read_csv(csv_path, chunksize=CHUNK_SIZE)):
        chunk = enrich(chunk)
        chunk.to_sql(
            TABLE_NAME,
            engine,
            if_exists="replace" if i == 0 else "append",
            index=True,
            index_label="id",
            method="multi",
            chunksize=5_000,
        )
        total_rows += len(chunk)
        logger.info("Chunk %d loaded (%d rows total)", i + 1, total_rows)

    create_indexes(engine)
    logger.info("Ingestion complete: %d rows in '%s'.", total_rows, TABLE_NAME)
    return total_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest creditcard.csv into PostgreSQL")
    parser.add_argument("--csv", default=os.getenv("DATA_PATH", "data/creditcard.csv"))
    args = parser.parse_args()

    try:
        ingest(args.csv)
    except Exception:
        logger.exception("Ingestion failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
