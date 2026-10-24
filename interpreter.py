"""
interpreter.py – Question interpretation service.

Converts a natural-language question into a validated analytical plan.

Architecture:
  1. Deterministic intent matching (fast, no LLM call) for well-known patterns.
  2. LLM-assisted interpretation for ambiguous questions.
  3. Plan validation against the allowlist in analytics.py.
  4. Fallback: "unsupported" response when interpretation fails.

The LLM is given:
  - A system prompt explaining the allowed operations and schema.
  - The user's question.
  - Instructions to return ONLY valid JSON.

The LLM must NOT calculate numbers. It only produces a plan.
The plan is then validated and executed deterministically.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional, Tuple

from analytics import ALLOWED_OPERATIONS, MONTH_NAMES, validate_plan, InvalidPlanError
from config import LLMConfig
from llm_providers import BaseLLMProvider, LLMError, get_provider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Month parsing helper
# ---------------------------------------------------------------------------
MONTH_MAP: dict[str, int] = {
    name.lower(): num for num, name in MONTH_NAMES.items()
}
# Also allow abbreviated months
MONTH_MAP.update({
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "may": 5, "jun": 6, "jul": 7, "aug": 8,
    "sep": 9, "oct": 10, "nov": 11, "dec": 12,
})


def _parse_month(text: str) -> Optional[int]:
    return MONTH_MAP.get(text.lower().strip())


def _parse_year(text: str) -> Optional[int]:
    m = re.search(r"\b(20\d{2})\b", text)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Deterministic intent patterns
# ---------------------------------------------------------------------------
def _match_deterministic(question: str) -> Optional[dict]:
    """
    Return a plan dict for well-known question patterns, or None if ambiguous.

    This avoids an LLM call for the most common questions.
    """
    q = question.lower().strip()

    # ── total revenue / sales ────────────────────────────────────────────
    if (
        re.search(r"\b(total|overall|lifetime|all)\s+(revenue|sales)\b|\b(revenue|sales)\s+(total|overall|lifetime)\b", q)
        and not re.search(r"\bby\b|\btrend\b|\bcompare\b|\bbetween\b", q)
    ):
        plan: dict = {"operation": "total_revenue", "filters": {}}
        # optional year
        y = _parse_year(q)
        if y:
            plan["filters"]["year"] = y
        # optional month
        for name, num in MONTH_MAP.items():
            if re.search(rf"\b{re.escape(name)}\b", q):
                plan["filters"]["month"] = num
                break
        return plan

    # ── monthly revenue / trend ─────────────────────────────────────────
    if re.search(r"\bmonthly\b|\btrend\b|\bper month\b|\bby month\b", q):
        plan = {"operation": "monthly_revenue", "filters": {}}
        y = _parse_year(q)
        if y:
            plan["filters"]["year"] = y
        return plan

    # ── top N products (e.g. "top 5 products by revenue") ────────────────
    m_topn = re.search(r"\btop\s+(\d+)\s+product", q)
    if m_topn:
        return {"operation": "top_products", "filters": {}, "sort_dir": "desc", "limit": int(m_topn.group(1))}

    # ── highest/best revenue product ─────────────────────────────────────
    if re.search(r"\b(highest|best|most)\b.*\bproduct\b|\bproduct\b.*\b(highest|best|most|generated)\b", q):
        plan: dict = {"operation": "top_products", "filters": {}, "sort_dir": "desc", "limit": 1}
        y = _parse_year(q)
        if y:
            plan["filters"]["year"] = y
        for name, num in MONTH_MAP.items():
            if re.search(rf"\b{re.escape(name)}\b", q):
                plan["filters"]["month"] = num
                break
        return plan

    # ── lowest revenue product ────────────────────────────────────────────
    if re.search(r"\b(lowest|worst|least|bottom)\b.*\bproduct\b|\bproduct\b.*\b(lowest|worst|least)\b", q):
        return {"operation": "top_products", "filters": {}, "sort_dir": "asc", "limit": 1}

    # ── revenue by product (generic grouping) ────────────────────────────
    if re.search(r"\bby product\b|\bper product\b|\bproduct.*revenue\b|\brevenue.*product\b", q):
        plan = {"operation": "revenue_by_product", "filters": {}, "sort_dir": "desc", "limit": 10}
        y = _parse_year(q)
        if y:
            plan["filters"]["year"] = y
        return plan

    # ── revenue by region ────────────────────────────────────────────────
    if re.search(r"\bby region\b|\bper region\b|\bregion.*revenue\b|\brevenue.*region\b", q):
        plan = {"operation": "revenue_by_region", "filters": {}, "sort_dir": "desc", "limit": 10}
        y = _parse_year(q)
        if y:
            plan["filters"]["year"] = y
        return plan

    # ── highest/best region ───────────────────────────────────────────────
    if re.search(r"\b(highest|best|most|top)\b.*\bregion\b|\bregion\b.*\b(highest|best|most)\b", q):
        plan = {"operation": "top_regions", "filters": {}, "sort_dir": "desc", "limit": 1}
        y = _parse_year(q)
        if y:
            plan["filters"]["year"] = y
        return plan

    # ── revenue by category ───────────────────────────────────────────────
    if re.search(r"\bby category\b|\bper category\b|\bcategory.*revenue\b|\brevenue.*categor", q):
        return {"operation": "revenue_by_category", "filters": {}, "sort_dir": "desc", "limit": 10}

    # ── revenue by channel ────────────────────────────────────────────────
    if re.search(r"\bonline.*retail\b|\bretail.*online\b|\bby channel\b|\bsales channel\b|\bchannel.*revenue\b", q):
        return {"operation": "revenue_by_channel", "filters": {}, "sort_dir": "desc", "limit": 10}

    # ── revenue comparison between two months ─────────────────────────────
    comp = re.search(r"\b(compare|between|vs\.?|versus)\b", q)
    if comp:
        found = []
        for name, num in MONTH_MAP.items():
            if re.search(rf"\b{re.escape(name)}\b", q):
                found.append(num)
        if len(found) >= 2:
            plan = {
                "operation": "revenue_comparison",
                "filters": {"month1": found[0], "month2": found[1]},
            }
            y = _parse_year(q)
            if y:
                plan["filters"]["year"] = y
            return plan

    # ── transaction count ─────────────────────────────────────────────────
    if re.search(r"\b(how many|count|number of)\b.*(transaction|order|sale)", q):
        plan = {"operation": "transaction_count", "filters": {}}
        if re.search(r"\bcompleted\b", q):
            plan["status"] = "Completed"
        elif re.search(r"\ball\b|\bany\b", q):
            plan["status"] = "all"
        return plan

    return None  # not matched – use LLM


# ---------------------------------------------------------------------------
# LLM-assisted interpretation
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are an analytical plan generator for a sales BI system.

Your ONLY job is to convert a natural-language question into a JSON plan.
You must NOT calculate any numbers. You must NOT invent data.

The database contains a table called `transactions` with these columns:
  Transaction_Id, Order_date (YYYY-MM-DD), Product, Category, Region,
  Sales_channel, Quantity, Unit_price, Discount_amount, Status, Revenue.

Allowed operations:
  total_revenue, revenue_by_product, revenue_by_region,
  revenue_by_category, revenue_by_channel, monthly_revenue,
  revenue_comparison, transaction_count, top_products, top_regions,
  product_revenue_in_region, revenue_in_period

Allowed filter keys:
  month (int 1-12), year (int), product (str), region (str),
  category (str), channel (str)
  For revenue_comparison only: month1 (int), month2 (int), year (int)

Other allowed keys:
  sort_dir: "desc" (default) or "asc"
  limit: integer 1-50 (default 10)
  status: "Completed" (default) or "all"

Rules:
- Return ONLY a single valid JSON object. No markdown, no explanation.
- If the question cannot be answered by these operations, return:
  {"operation": "unsupported", "reason": "brief explanation"}
- Use 2026 as the default year if the user mentions a month without a year.
- months are integers: January=1, February=2, ..., December=12.

Examples:
  Q: What is the total revenue?
  A: {"operation":"total_revenue","filters":{}}

  Q: Which product had the highest revenue in January 2026?
  A: {"operation":"top_products","filters":{"month":1,"year":2026},"sort_dir":"desc","limit":1}

  Q: Compare revenue between January and February
  A: {"operation":"revenue_comparison","filters":{"month1":1,"month2":2,"year":2026}}

  Q: Show monthly trend
  A: {"operation":"monthly_revenue","filters":{}}
"""


def _call_llm(question: str, provider: BaseLLMProvider) -> dict:
    """Call the LLM and parse the JSON plan from its response."""
    raw = provider.complete(SYSTEM_PROMPT, question)
    # Strip markdown fences if present
    cleaned = re.sub(r"```(?:json)?", "", raw, flags=re.IGNORECASE).replace("```", "").strip()
    # Find first JSON object
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        raise LLMError(f"LLM returned no JSON object. Raw: {cleaned[:200]!r}")
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError as exc:
        raise LLMError(f"LLM returned malformed JSON: {exc}. Raw: {cleaned[:200]!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
InterpretResult = Tuple[Optional[dict], str, str]
# Returns: (plan_or_None, explanation, path_used)
#   path_used: "deterministic" | "llm" | "unsupported" | "error"


def interpret_question(
    question: str,
    llm_cfg: Optional[LLMConfig] = None,
) -> InterpretResult:
    """
    Interpret a natural-language question and return a validated plan.

    Returns:
        (plan, explanation, path)

    If interpretation fails or is unsupported, plan is None.
    """
    question = question.strip()

    if not question:
        return None, "Please enter a question.", "error"
    if len(question) > 1000:
        return None, "Question is too long (max 1000 characters).", "error"

    # ── Step 1: deterministic matching ───────────────────────────────────
    det_plan = _match_deterministic(question)
    if det_plan is not None:
        try:
            validated = validate_plan(det_plan)
            logger.info("Deterministic match for: %r → %s", question, validated["operation"])
            return validated, "", "deterministic"
        except InvalidPlanError as e:
            logger.warning("Deterministic plan invalid: %s", e)
            # Fall through to LLM

    # ── Step 2: LLM-assisted interpretation ─────────────────────────────
    if llm_cfg is None or not llm_cfg.is_valid():
        return (
            None,
            "This question requires AI interpretation, but no LLM is configured. "
            "Please set LLM_PROVIDER and LLM_API_KEY in your .env file or the settings panel.",
            "error",
        )

    try:
        provider = get_provider(llm_cfg)
        raw_plan = _call_llm(question, provider)

        if raw_plan.get("operation") == "unsupported":
            reason = raw_plan.get("reason", "This question type is not supported.")
            return None, reason, "unsupported"

        validated = validate_plan(raw_plan)
        logger.info("LLM plan for: %r → %s", question, validated["operation"])
        return validated, "", "llm"

    except InvalidPlanError as e:
        return None, f"The AI returned an invalid plan: {e}", "error"
    except LLMError as e:
        if not e.retryable:
            return None, str(e), "error"
        return None, f"AI service temporarily unavailable: {e}", "error"
    except Exception as e:
        logger.error("Unexpected interpretation error: %s", e, exc_info=True)
        return None, "An unexpected error occurred during question interpretation.", "error"


# ---------------------------------------------------------------------------
# Natural-language explanation generator
# ---------------------------------------------------------------------------
EXPLAIN_SYSTEM = """You are a concise business analyst explaining query results.
Given a question and its results (numbers from the database), write a single short
paragraph in plain English. Do NOT invent numbers. Use only the data provided.
Do not use bullet points. Keep it under 80 words."""


def generate_explanation(
    question: str,
    result: dict,
    llm_cfg: Optional[LLMConfig] = None,
) -> str:
    """
    Generate a short natural-language explanation of the analytics result.

    Falls back to a structured summary if LLM is unavailable.
    """
    rows  = result.get("rows", [])
    cols  = result.get("columns", [])
    op    = result.get("operation", "")

    # ── Deterministic fallback explanation ────────────────────────────────
    if not rows:
        return "No data matches the specified criteria."

    if op == "total_revenue":
        val = rows[0][0]
        return f"The total revenue is **£{val:,.2f}**." if val is not None else "No revenue data available."

    if op in ("top_products", "revenue_by_product") and rows:
        top = rows[0]
        return f"**{top[0]}** generated the highest revenue at **£{top[1]:,.2f}**."

    if op in ("top_regions", "revenue_by_region") and rows:
        top = rows[0]
        return f"The **{top[0]}** region generated the highest revenue at **£{top[1]:,.2f}**."

    if op == "revenue_comparison" and len(rows) >= 2:
        r1, r2 = rows[0], rows[1]
        v1 = r1[1] or 0
        v2 = r2[1] or 0
        diff = abs(v1 - v2)
        higher = r1[0] if v1 >= v2 else r2[0]
        return (
            f"**{r1[0]}** had revenue of **£{v1:,.2f}** and "
            f"**{r2[0]}** had **£{v2:,.2f}**. "
            f"**{higher}** was higher by **£{diff:,.2f}**."
        )

    if op == "transaction_count" and rows:
        return f"There are **{rows[0][0]:,}** transactions matching the criteria."

    # ── LLM explanation ───────────────────────────────────────────────────
    if llm_cfg and llm_cfg.is_valid():
        try:
            provider = get_provider(llm_cfg)
            data_str = f"Question: {question}\n\nColumns: {cols}\n\nRows (first 10): {rows[:10]}"
            text = provider.complete(EXPLAIN_SYSTEM, data_str)
            return text.strip()
        except LLMError as e:
            logger.warning("Explanation LLM failed: %s", e)

    # ── Generic fallback ──────────────────────────────────────────────────
    return f"Query returned {len(rows)} row(s) for operation '{op}'."
