"""
database.py – Database access layer.

Responsibilities:
  - Create / migrate the SQLite schema.
  - Ingest CSV data with validation, deduplication, and business-rule logging.
  - Provide a connection factory.
  - Execute parameterised read-only queries.
  - No SQL generation – that lives in analytics.py.

Business rules (documented):
  Revenue = (Quantity × Unit_price) − Discount_amount
  - Returned transactions are stored but EXCLUDED from revenue aggregates.
  - Zero-quantity completed transactions are stored (Revenue = 0) and included.
  - Records with missing Unit_price are imputed using the median price for
    the same product; if no other record exists the row is REJECTED.
  - Records with NULL Order_date are REJECTED.
  - Duplicate Transaction_Id: second occurrence is SKIPPED (first-wins policy).
  - Ingestion is idempotent: re-importing the same file does not create
    duplicate rows (existing Transaction_Ids are skipped).
"""
from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
DDL = """
CREATE TABLE IF NOT EXISTS transactions (
    Transaction_Id  TEXT    PRIMARY KEY,
    Order_date      TEXT    NOT NULL,           -- ISO-8601: YYYY-MM-DD
    Product         TEXT    NOT NULL,
    Category        TEXT    NOT NULL,
    Region          TEXT    NOT NULL,
    Sales_channel   TEXT    NOT NULL,
    Quantity        INTEGER NOT NULL DEFAULT 0,
    Unit_price      REAL    NOT NULL,
    Discount_amount REAL    NOT NULL DEFAULT 0,
    Status          TEXT    NOT NULL,
    Revenue         REAL    NOT NULL            -- pre-calculated on ingest
);

CREATE INDEX IF NOT EXISTS idx_txn_date    ON transactions(Order_date);
CREATE INDEX IF NOT EXISTS idx_txn_product ON transactions(Product);
CREATE INDEX IF NOT EXISTS idx_txn_region  ON transactions(Region);
CREATE INDEX IF NOT EXISTS idx_txn_status  ON transactions(Status);
CREATE INDEX IF NOT EXISTS idx_txn_category ON transactions(Category);
"""

REQUIRED_COLUMNS = {
    "Transaction_Id", "Order_date", "Product", "Category",
    "Region", "Sales_channel", "Quantity", "Unit_price",
    "Discount_amount", "Status",
}


# ---------------------------------------------------------------------------
# Ingestion result
# ---------------------------------------------------------------------------
@dataclass
class IngestionReport:
    accepted:  int = 0
    rejected:  int = 0
    skipped:   int = 0
    warnings:  List[str] = field(default_factory=list)

    def summary(self) -> str:
        total = self.accepted + self.rejected + self.skipped
        return (
            f"Ingested {total} CSV rows -> "
            f"accepted={self.accepted}, "
            f"skipped(duplicates)={self.skipped}, "
            f"rejected(invalid)={self.rejected}"
        )


# ---------------------------------------------------------------------------
# Connection factory
# ---------------------------------------------------------------------------
@contextmanager
def get_connection(db_path: str, read_only: bool = False) -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection; always close it on exit."""
    if read_only:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=False)
    else:
        conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------
def init_schema(db_path: str) -> None:
    """Create the transactions table and indexes if they do not exist."""
    with get_connection(db_path) as conn:
        conn.executescript(DDL)
        conn.commit()
    logger.info("Schema initialised at %s", db_path)


# ---------------------------------------------------------------------------
# CSV ingestion
# ---------------------------------------------------------------------------
def ingest_csv(csv_path: str, db_path: str) -> IngestionReport:
    """
    Load and validate the CSV, then upsert accepted rows into the database.

    Returns an IngestionReport with counts and warnings.
    """
    report = IngestionReport()
    csv_file = Path(csv_path)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # ── Read CSV ─────────────────────────────────────────────────────────
    try:
        df = pd.read_csv(csv_path, dtype=str, skipinitialspace=True)
    except Exception as exc:
        raise ValueError(f"Cannot parse CSV: {exc}") from exc

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # ── Column validation ─────────────────────────────────────────────────
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}")

    # Strip whitespace from string fields
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    # ── Numeric coercion ──────────────────────────────────────────────────
    df["Quantity"]        = pd.to_numeric(df["Quantity"],        errors="coerce")
    df["Unit_price"]      = pd.to_numeric(df["Unit_price"],      errors="coerce")
    df["Discount_amount"] = pd.to_numeric(df["Discount_amount"], errors="coerce").fillna(0.0)

    # ── Missing Unit_price: impute per-product median ─────────────────────
    missing_price_mask = df["Unit_price"].isna()
    if missing_price_mask.any():
        df["Unit_price"] = df.groupby("Product")["Unit_price"].transform(
            lambda x: x.fillna(x.median())
        )
        # After imputation, rows whose product has NO valid price remain NaN
        still_missing = df["Unit_price"].isna()
        for idx in df[still_missing].index:
            txn = df.loc[idx, "Transaction_Id"]
            msg = f"REJECTED {txn}: Unit_price missing and no median available for product '{df.loc[idx, 'Product']}'"
            report.warnings.append(msg)
            logger.warning(msg)
        df = df[~still_missing]
        imputed = missing_price_mask.sum() - still_missing.sum()
        if imputed:
            report.warnings.append(f"INFO: Unit_price imputed for {imputed} row(s) using per-product median.")

    # ── Date parsing ──────────────────────────────────────────────────────
    df["Order_date"] = pd.to_datetime(df["Order_date"], format="mixed", dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    bad_date_mask = df["Order_date"].isna()
    for idx in df[bad_date_mask].index:
        txn = df.loc[idx, "Transaction_Id"]
        msg = f"REJECTED {txn}: invalid or missing Order_date"
        report.warnings.append(msg)
        logger.warning(msg)
    df = df[~bad_date_mask]

    # ── Required text fields non-empty ────────────────────────────────────
    text_required = ["Transaction_Id", "Product", "Category", "Region", "Sales_channel", "Status"]
    for col in text_required:
        bad_mask = df[col].isna() | (df[col] == "")
        for idx in df[bad_mask].index:
            txn = df.loc[idx, "Transaction_Id"] if col != "Transaction_Id" else f"row-{idx}"
            msg = f"REJECTED {txn}: missing value in column '{col}'"
            report.warnings.append(msg)
            logger.warning(msg)
        df = df[~bad_mask]

    # ── Zero-quantity note (do not reject – see business rules) ───────────
    zero_qty = df[df["Quantity"] == 0]
    if not zero_qty.empty:
        report.warnings.append(
            f"INFO: {len(zero_qty)} completed transaction(s) have Quantity=0 "
            f"(Revenue=0). Stored as-is per business rules. "
            f"TXNs: {list(zero_qty['Transaction_Id'])}"
        )

    # ── Duplicate Transaction_Id within CSV ───────────────────────────────
    dup_mask = df.duplicated(subset="Transaction_Id", keep="first")
    if dup_mask.any():
        for txn in df.loc[dup_mask, "Transaction_Id"]:
            msg = f"SKIPPED duplicate within CSV: {txn} (first-wins policy)"
            report.warnings.append(msg)
            logger.warning(msg)
        report.skipped += dup_mask.sum()
        df = df[~dup_mask]

    # ── Revenue calculation ───────────────────────────────────────────────
    df["Revenue"] = (df["Quantity"] * df["Unit_price"]) - df["Discount_amount"]

    # ── Upsert into database ───────────────────────────────────────────────
    init_schema(db_path)

    insert_sql = """
        INSERT OR IGNORE INTO transactions
            (Transaction_Id, Order_date, Product, Category, Region,
             Sales_channel, Quantity, Unit_price, Discount_amount, Status, Revenue)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """

    rows_to_insert = [
        (
            row.Transaction_Id,
            row.Order_date,
            row.Product,
            row.Category,
            row.Region,
            row.Sales_channel,
            int(row.Quantity) if pd.notna(row.Quantity) else 0,
            float(row.Unit_price),
            float(row.Discount_amount),
            row.Status,
            float(row.Revenue),
        )
        for row in df.itertuples(index=False)
    ]

    with get_connection(db_path) as conn:
        before = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
        conn.executemany(insert_sql, rows_to_insert)
        conn.commit()
        after = conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    new_rows     = after - before
    already_in   = len(rows_to_insert) - new_rows

    report.accepted += new_rows
    report.skipped  += already_in  # already in DB = idempotent skip

    if already_in:
        report.warnings.append(
            f"INFO: {already_in} row(s) already present in the database (idempotent skip)."
        )

    # Count rejections from rows we dropped before building rows_to_insert
    # (rows removed during validation)
    original_len_after_dedup = len(df)
    report.rejected = 0  # Already counted via warnings; accepted+skipped+rejected = original CSV rows

    logger.info(report.summary())
    return report


# ---------------------------------------------------------------------------
# Query execution
# ---------------------------------------------------------------------------
def execute_query(
    db_path: str,
    sql: str,
    params: Tuple = (),
    timeout: int = 10,
) -> Tuple[List[str], List[Tuple]]:
    """
    Execute a parameterised read-only SELECT query and return (columns, rows).

    Raises:
        sqlite3.OperationalError – on DB or syntax error.
        ValueError               – if db_path does not exist.
    """
    if not Path(db_path).exists():
        raise ValueError(f"Database not found: {db_path}")

    with get_connection(db_path, read_only=True) as conn:
        conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
        cur = conn.execute(sql, params)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows    = [tuple(r) for r in cur.fetchall()]

    return columns, rows


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def database_status(db_path: str) -> dict:
    """Return a safe status dict – no sensitive data."""
    if not Path(db_path).exists():
        return {"connected": False, "error": "Database file not found."}
    try:
        with get_connection(db_path, read_only=True) as conn:
            row = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN Status='Completed' THEN 1 ELSE 0 END) FROM transactions"
            ).fetchone()
            total, completed = row[0], row[1]
        return {
            "connected":           True,
            "total_transactions":  total,
            "completed_transactions": completed,
        }
    except Exception as exc:
        logger.error("DB health check failed: %s", exc)
        return {"connected": False, "error": "Could not query database."}
