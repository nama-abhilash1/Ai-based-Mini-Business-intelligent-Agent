# AI-Assisted Development Challenge Documentation

## 1. Architecture Overview

This application implements a **Plan-Based Analytical Engine with Deterministic Execution**:

```
User Query (Natural Language)
        │
        ▼
┌─────────────────────────────────┐
│       Streamlit Frontend        │  Validates input, provides session configuration,
│           (app.py)              │  renders KPI cards, interactive charts, and tables
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│     Question Interpreter        │  1. Deterministic Regex Matcher (Zero-latency, $0 cost)
│        (interpreter.py)         │  2. LLM Intent Interpreter (Fallback for complex phrasing)
└───────────────┬─────────────────┘     Produces structured, validated analytical plan JSON
                │
                ▼
┌─────────────────────────────────┐
│       Analytics Engine          │  Enforces strict operation allowlist & parameter types.
│        (analytics.py)           │  Executes parameterised, pre-vetted SQL queries.
└───────────────┬─────────────────┘  Calculates accurate revenue: (Quantity × Unit_Price) - Discount
                │
                ▼
┌─────────────────────────────────┐
│       Database Layer            │  SQLite database with indexes, schema validation,
│        (database.py)            │  idempotent CSV ingestion, and read-only query safety
└───────────────┬─────────────────┘
                │
                ▼
┌─────────────────────────────────┐
│    Explainability Synthesizer   │  Translates query result tuples into concise, executive-ready
│   (interpreter.py + LLM/Rules)  │  business insights strictly bound to verified facts
└─────────────────────────────────┘
```

### Key Architectural Decisions
* **Deterministic Execution over Unconstrained Text-to-SQL**: Instead of letting an LLM write raw SQL directly to run on the database (which causes syntax errors, hallucinations, security vulnerabilities, and metric calculation discrepancies), the LLM acts solely as a query planner producing a structured JSON schema. The backend maps approved plans to vetted, parameterised SQL queries.
* **Separation of Concerns**: UI rendering (`app.py`), question interpretation (`interpreter.py`), analytical calculations (`analytics.py`), database interactions (`database.py`), and LLM provider abstractions (`llm_providers.py`) are strictly decoupled.
* **Provider Agnostic**: Native support for Google Gemini, OpenAI, Anthropic Claude, and Groq Cloud via a unified adapter with automatic timeout handling, temperature calibration, and graceful fallback.
* **Strict Business Logic Compliance**: Excludes `Status = 'Returned'` from revenue totals; accurately calculates revenue as `(Quantity * Unit_price) - Discount_amount`; imputes missing product unit prices using median category prices; deduplicates transactions during ingestion.

---

## 2. AI Tools Used

* **Google Gemini (`gemini-1.5-flash` / `gemini-2.0-flash`)**: Used for zero-shot natural language intent parsing into structured JSON plans and generating natural language summaries.
* **Groq (`llama-3.3-70b-versatile`, `llama-3.1-8b-instant`)**: High-speed, ultra-low latency inference backend.
* **OpenAI (`gpt-4o-mini`) & Anthropic (`claude-3-haiku`)**: Supported alternative LLM backends via provider abstraction.
* **AI-Assisted Pair Programming**:
  * Used to scaffold and refine the regex-based deterministic intent matcher.
  * Used to structure property-based and edge-case pytest fixtures across database, analytics, interpreter, and security layers.
  * Used for modern dark-mode UI styling and CSS glassmorphism components in Streamlit.

---

## 3. Challenges & Debugging

1. **Revenue Calculation Inconsistencies**:
   * *Problem*: Early implementations computed revenue simply as `Quantity * Unit_price` or included returned orders, misrepresenting actual business earnings.
   * *Fix*: Centralized business logic in `analytics.py` ensuring all queries compute `SUM((Quantity * Unit_price) - Discount_amount)` and filter `WHERE Status != 'Returned'` unless returned transactions are explicitly queried.

2. **LLM Markdown Wrapping and Uncontrolled SQL Generation**:
   * *Problem*: Freeform text-to-SQL prompts frequently produced markdown fencing (````sql ... ````), destructive statements, or hallucinated schema column names.
   * *Fix*: Switched to structured plan JSON extraction with schema enforcement (`validate_plan`). The analytical engine controls all SQL templates, completely eliminating SQL injection and syntax failures.

3. **Database Idempotency and Dirty Data**:
   * *Problem*: Repeated setup runs duplicated rows; dirty data contained European date formats (`DD-MM-YYYY` vs `YYYY-MM-DD`), missing prices, and empty rows.
   * *Fix*: Implemented robust ingestion in `database.py` with multi-format date coercion, median price imputation for missing values, primary key deduplication, and an audit report.

4. **Cross-Platform Encoding on Windows**:
   * *Problem*: Unicode glyphs (arrows, checkmarks) in terminal output caused `UnicodeEncodeError` under Windows default `cp1252` encoding.
   * *Fix*: Standardized CLI messages to clean ASCII labels `[OK]`, `[INFO]`, `->` and enabled UTF-8 compatibility.

---

## 4. Future Improvements

Given further development cycles, planned enhancements include:
* **Semantic Caching Layer**: Cache frequent user question embeddings to instantly retrieve analytical plans without invoking LLM APIs.
* **Self-Healing Clarification Engine**: When user queries are ambiguous (e.g., "compare products" without specifying metrics or periods), prompt interactive follow-up selection directly within the Streamlit UI.
* **Dynamic Multi-CSV Ingestion**: Allow users to drag-and-drop multiple transaction tables, auto-infer relationships, and dynamically join them into the analytics catalog.
* **Automated PDF / Executive Export**: Generate downloadable branded executive PDF reports with embedded charts and AI commentary.
