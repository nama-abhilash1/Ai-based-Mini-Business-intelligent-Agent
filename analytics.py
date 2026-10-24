"""
analytics.py – Deterministic analytics service.

All numerical results come from this module only.
The LLM never calculates sales figures.

Supported operations (allowlist):
  - total_revenue
  - revenue_by_product
  - revenue_by_region
  - revenue_by_category
  - revenue_by_channel
  - monthly_revenue
  - revenue_comparison
  - transaction_count
  - top_products
  - top_regions
  - product_revenue_in_region
  - revenue_in_period

Business rules applied:
  - Returned transactions are EXCLUDED from revenue aggregates unless
    the caller explicitly requests all statuses.
  - Zero-quantity completed transactions contribute Revenue = 0.
  - Revenue = (Quantity × Unit_price) − Discount_amount (pre-calculated on ingest).

Analytical plan schema:
{
  "operation":  str,       # one of ALLOWED_OPERATIONS
  "filters":    dict,      # optional: month, year, region, product, category, channel
  "dimension":  str,       # optional grouping column
  "sort_dir":   str,       # "desc" | "asc"
  "limit":      int,       # 1-50
  "status":     str        # "Completed" | "all" – default "Completed"
}
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from database import execute_query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALLOWED_OPERATIONS = {
    "total_revenue",
    "revenue_by_product",
    "revenue_by_region",
    "revenue_by_category",
    "revenue_by_channel",
    "monthly_revenue",
    "revenue_comparison",
    "transaction_count",
    "top_products",
    "top_regions",
    "product_revenue_in_region",
    "revenue_in_period",
}

ALLOWED_DIMENSIONS = {
    "product":       "Product",
    "region":        "Region",
    "category":      "Category",
    "channel":       "Sales_channel",
    "month":         "strftime('%Y-%m', Order_date)",
    "status":        "Status",
}

ALLOWED_SORT_DIRS = {"asc", "desc"}
MONTH_NAMES = {
    1: "January", 2: "February",  3: "March",    4: "April",
    5: "May",     6: "June",      7: "July",     8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


# ---------------------------------------------------------------------------
# Plan validation
# ---------------------------------------------------------------------------
class InvalidPlanError(ValueError):
    pass


def validate_plan(plan: dict) -> dict:
    """
    Validate and normalise an analytical plan dict.

    Raises InvalidPlanError with a descriptive message on any problem.
    Returns the normalised plan.
    """
    if not isinstance(plan, dict):
        raise InvalidPlanError("Plan must be a JSON object.")

    operation = plan.get("operation", "")
    if operation not in ALLOWED_OPERATIONS:
        raise InvalidPlanError(
            f"Unsupported operation '{operation}'. "
            f"Allowed: {sorted(ALLOWED_OPERATIONS)}"
        )

    # Filters
    filters = plan.get("filters", {})
    if not isinstance(filters, dict):
        raise InvalidPlanError("'filters' must be a JSON object.")

    if "month" in filters:
        m = filters["month"]
        if not isinstance(m, int) or not (1 <= m <= 12):
            raise InvalidPlanError(f"'filters.month' must be an integer 1-12, got {m!r}.")

    if "year" in filters:
        y = filters["year"]
        if not isinstance(y, int) or not (2000 <= y <= 2100):
            raise InvalidPlanError(f"'filters.year' must be a 4-digit year, got {y!r}.")

    if "limit" in plan:
        lim = plan["limit"]
        if not isinstance(lim, int) or not (1 <= lim <= 50):
            raise InvalidPlanError(f"'limit' must be an integer 1-50, got {lim!r}.")

    sort_dir = plan.get("sort_dir", "desc").lower()
    if sort_dir not in ALLOWED_SORT_DIRS:
        raise InvalidPlanError(f"'sort_dir' must be 'asc' or 'desc', got {sort_dir!r}.")

    status = plan.get("status", "Completed")
    if status not in ("Completed", "all"):
        raise InvalidPlanError(f"'status' must be 'Completed' or 'all', got {status!r}.")

    # Return normalised copy
    return {
        "operation": operation,
        "filters":   filters,
        "sort_dir":  sort_dir,
        "limit":     plan.get("limit", 10),
        "status":    status,
        "dimension": plan.get("dimension", ""),
    }


# ---------------------------------------------------------------------------
# Query builder helpers
# ---------------------------------------------------------------------------
def _status_clause(status: str, params: list) -> str:
    if status == "Completed":
        params.append("Completed")
        return " AND Status = ?"
    return ""  # "all" – no filter


def _date_clauses(filters: dict, params: list) -> str:
    clauses = []
    if "year" in filters:
        clauses.append("strftime('%Y', Order_date) = ?")
        params.append(str(filters["year"]))
    if "month" in filters and "year" in filters:
        clauses.append("strftime('%m', Order_date) = ?")
        params.append(f"{filters['month']:02d}")
    elif "month" in filters:
        # Month without year – warn but still filter
        clauses.append("strftime('%m', Order_date) = ?")
        params.append(f"{filters['month']:02d}")
    return (" AND " + " AND ".join(clauses)) if clauses else ""


def _entity_clauses(filters: dict, params: list) -> str:
    clauses = []
    if "product" in filters:
        clauses.append("LOWER(Product) = LOWER(?)")
        params.append(filters["product"])
    if "region" in filters:
        clauses.append("LOWER(Region) = LOWER(?)")
        params.append(filters["region"])
    if "category" in filters:
        clauses.append("LOWER(Category) = LOWER(?)")
        params.append(filters["category"])
    if "channel" in filters:
        clauses.append("LOWER(Sales_channel) = LOWER(?)")
        params.append(filters["channel"])
    return (" AND " + " AND ".join(clauses)) if clauses else ""


# ---------------------------------------------------------------------------
# Analytics result
# ---------------------------------------------------------------------------
AnalyticsResult = Dict[str, Any]


def _make_result(operation: str, columns: List[str], rows: List[Tuple], meta: dict = None) -> AnalyticsResult:
    return {
        "operation": operation,
        "columns":   columns,
        "rows":      rows,
        "meta":      meta or {},
    }


# ---------------------------------------------------------------------------
# Individual analytics functions
# ---------------------------------------------------------------------------

def _total_revenue(plan: dict, db_path: str) -> AnalyticsResult:
    params: list = []
    status_c = _status_clause(plan["status"], params)
    date_c   = _date_clauses(plan["filters"],  params)
    entity_c = _entity_clauses(plan["filters"], params)

    sql = f"SELECT SUM(Revenue) AS total_revenue FROM transactions WHERE 1=1{status_c}{date_c}{entity_c}"
    cols, rows = execute_query(db_path, sql, tuple(params))
    return _make_result("total_revenue", cols, rows,
                        {"description": "Sum of Revenue for completed transactions."})


def _revenue_by_dimension(plan: dict, db_path: str, dim_col: str, dim_label: str) -> AnalyticsResult:
    params: list = []
    status_c = _status_clause(plan["status"], params)
    date_c   = _date_clauses(plan["filters"],  params)
    entity_c = _entity_clauses(plan["filters"], params)
    sort     = "ASC" if plan["sort_dir"] == "asc" else "DESC"
    limit    = plan["limit"]

    sql = (
        f"SELECT {dim_col} AS {dim_label}, SUM(Revenue) AS revenue "
        f"FROM transactions WHERE 1=1{status_c}{date_c}{entity_c} "
        f"GROUP BY {dim_col} ORDER BY revenue {sort} LIMIT ?"
    )
    params.append(limit)
    cols, rows = execute_query(db_path, sql, tuple(params))
    return _make_result(plan["operation"], cols, rows,
                        {"description": f"Revenue grouped by {dim_label}."})


def _monthly_revenue(plan: dict, db_path: str) -> AnalyticsResult:
    params: list = []
    status_c = _status_clause(plan["status"], params)
    entity_c = _entity_clauses(plan["filters"], params)

    if "year" in plan["filters"]:
        params.append(str(plan["filters"]["year"]))
        year_c = " AND strftime('%Y', Order_date) = ?"
    else:
        year_c = ""

    sql = (
        "SELECT strftime('%Y-%m', Order_date) AS month, SUM(Revenue) AS revenue "
        f"FROM transactions WHERE 1=1{status_c}{year_c}{entity_c} "
        "GROUP BY month ORDER BY month ASC"
    )
    cols, rows = execute_query(db_path, sql, tuple(params))
    return _make_result("monthly_revenue", cols, rows,
                        {"description": "Monthly revenue trend."})


def _revenue_comparison(plan: dict, db_path: str) -> AnalyticsResult:
    """Compare revenue between two months."""
    filters = plan["filters"]
    month1  = filters.get("month1")
    month2  = filters.get("month2")
    year    = filters.get("year")

    if month1 is None or month2 is None:
        raise InvalidPlanError(
            "revenue_comparison requires 'filters.month1' and 'filters.month2' (integers 1-12)."
        )
    for m in (month1, month2):
        if not isinstance(m, int) or not (1 <= m <= 12):
            raise InvalidPlanError(f"month value {m!r} must be an integer 1-12.")

    rows_out = []
    cols_out = ["period", "revenue"]

    for m in (month1, month2):
        p: list = []
        status_c = _status_clause(plan["status"], p)
        p.append(f"{m:02d}")
        year_c = ""
        if year:
            p.append(str(year))
            year_c = " AND strftime('%Y', Order_date) = ?"

        sql = (
            "SELECT SUM(Revenue) FROM transactions "
            f"WHERE strftime('%m', Order_date) = ?{year_c}{status_c}"
        )
        _, rs = execute_query(db_path, sql, tuple(p))
        val   = rs[0][0] if rs else None
        label = f"{MONTH_NAMES.get(m, str(m))}" + (f" {year}" if year else "")
        rows_out.append((label, val))

    return _make_result("revenue_comparison", cols_out, rows_out,
                        {"description": "Revenue comparison between two months."})


def _transaction_count(plan: dict, db_path: str) -> AnalyticsResult:
    params: list = []
    status_c = _status_clause(plan["status"], params)
    date_c   = _date_clauses(plan["filters"],  params)
    entity_c = _entity_clauses(plan["filters"], params)

    sql = f"SELECT COUNT(*) AS count FROM transactions WHERE 1=1{status_c}{date_c}{entity_c}"
    cols, rows = execute_query(db_path, sql, tuple(params))
    return _make_result("transaction_count", cols, rows,
                        {"description": "Number of transactions matching the criteria."})


def _product_revenue_in_region(plan: dict, db_path: str) -> AnalyticsResult:
    """Revenue for a specific product filtered by region."""
    plan2 = dict(plan)
    plan2["operation"] = "revenue_by_product"
    return _revenue_by_dimension(plan2, db_path, "Product", "product")


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
OPERATION_MAP = {
    "total_revenue":           _total_revenue,
    "revenue_by_product":      lambda p, d: _revenue_by_dimension(p, d, "Product",       "product"),
    "revenue_by_region":       lambda p, d: _revenue_by_dimension(p, d, "Region",        "region"),
    "revenue_by_category":     lambda p, d: _revenue_by_dimension(p, d, "Category",      "category"),
    "revenue_by_channel":      lambda p, d: _revenue_by_dimension(p, d, "Sales_channel", "channel"),
    "monthly_revenue":         _monthly_revenue,
    "revenue_comparison":      _revenue_comparison,
    "transaction_count":       _transaction_count,
    "top_products":            lambda p, d: _revenue_by_dimension(p, d, "Product",  "product"),
    "top_regions":             lambda p, d: _revenue_by_dimension(p, d, "Region",   "region"),
    "product_revenue_in_region": _product_revenue_in_region,
    "revenue_in_period":       _total_revenue,
}


def execute_plan(plan: dict, db_path: str) -> AnalyticsResult:
    """
    Validate a plan and execute the corresponding analytics query.

    This is the only entry point for analytics from outside this module.
    """
    validated = validate_plan(plan)
    op        = validated["operation"]
    fn        = OPERATION_MAP.get(op)
    if fn is None:
        raise InvalidPlanError(f"No handler for operation '{op}'.")
    logger.info("Executing plan: %s", validated)
    result = fn(validated, db_path)
    logger.info("Plan %s returned %d row(s).", op, len(result.get("rows", [])))
    return result


# ---------------------------------------------------------------------------
# Dashboard summary (pre-calculated for the overview cards)
# ---------------------------------------------------------------------------
def get_dashboard_summary(db_path: str) -> dict:
    """Return key metrics for the overview dashboard. Never raises."""
    try:
        _, r = execute_query(
            db_path,
            "SELECT SUM(Revenue), COUNT(*) FROM transactions WHERE Status='Completed'",
        )
        total_rev  = r[0][0] if r else None
        total_txns = r[0][1] if r else 0

        _, r2 = execute_query(
            db_path,
            "SELECT Product, SUM(Revenue) AS rev FROM transactions WHERE Status='Completed' "
            "GROUP BY Product ORDER BY rev DESC LIMIT 1",
        )
        top_product = r2[0][0] if r2 else "N/A"

        _, r3 = execute_query(
            db_path,
            "SELECT Region, SUM(Revenue) AS rev FROM transactions WHERE Status='Completed' "
            "GROUP BY Region ORDER BY rev DESC LIMIT 1",
        )
        top_region = r3[0][0] if r3 else "N/A"

        _, r4 = execute_query(
            db_path,
            "SELECT MIN(Order_date), MAX(Order_date) FROM transactions",
        )
        date_range = (r4[0][0], r4[0][1]) if r4 else ("N/A", "N/A")

        return {
            "total_revenue":       total_rev,
            "total_transactions":  total_txns,
            "top_product":         top_product,
            "top_region":          top_region,
            "date_from":           date_range[0],
            "date_to":             date_range[1],
        }
    except Exception as exc:
        logger.error("Dashboard summary failed: %s", exc)
        return {
            "total_revenue":      None,
            "total_transactions": None,
            "top_product":        "N/A",
            "top_region":         "N/A",
            "date_from":          "N/A",
            "date_to":            "N/A",
        }
