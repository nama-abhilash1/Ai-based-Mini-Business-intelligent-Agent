# AI Business Intelligence Assistant

A production-ready, interview-ready **AI-powered BI tool** that lets business users ask natural-language questions about sales data and receive deterministic, data-backed answers — with optional AI interpretation via OpenAI, Google Gemini, or Anthropic Claude.

---

## Table of Contents

1. [What It Does](#what-it-does)
2. [Architecture and Data Flow](#architecture-and-data-flow)
3. [Technology Stack](#technology-stack)
4. [Database Schema](#database-schema)
5. [CSV Ingestion and Data Rules](#csv-ingestion-and-data-rules)
6. [Revenue Calculation Rules](#revenue-calculation-rules)
7. [Supported Analytical Questions](#supported-analytical-questions)
8. [LLM Provider Configuration](#llm-provider-configuration)
9. [API / Analytical Operations](#api--analytical-operations)
10. [Installation](#installation)
11. [Environment Variables](#environment-variables)
12. [Running the Application](#running-the-application)
13. [Running Automated Tests](#running-automated-tests)
14. [Security Considerations](#security-considerations)
15. [Performance Notes](#performance-notes)
16. [Known Limitations](#known-limitations)
17. [Future Improvements](#future-improvements)

---

## What It Does

The AI BI Assistant allows a non-technical business user to:

- Ask plain-English questions about sales data (e.g. *"Which product generated the highest revenue?"*)
- View charts, tables, and natural-language explanations of results
- Explore an auto-populated dashboard with key KPIs
- Load or reload a CSV file via the sidebar
- Switch between LLM providers (OpenAI, Gemini, Anthropic) through configuration

**All numerical results come from deterministic SQL queries.** The LLM is used only for question interpretation and generating plain-English explanations — it never calculates sales figures.

---

## Architecture and Data Flow

```
User enters question
        │
        ▼
┌─────────────────┐
│  app.py         │  Streamlit UI — input validation, display
│  (Frontend)     │
└────────┬────────┘
         │ question string
         ▼
┌─────────────────────────┐
│  interpreter.py          │  Step 1: Deterministic regex match
│  (Question Interpreter)  │  Step 2: LLM-assisted interpretation (if needed)
│                          │  → Returns validated analytical plan dict
└────────┬────────────────┘
         │ validated plan
         ▼
┌─────────────────────────┐
│  analytics.py            │  Plan validated against allowlist
│  (Analytics Service)     │  Deterministic SQL query executed
│                          │  → Returns columns + rows (real data)
└────────┬────────────────┘
         │ result dict
         ▼
┌─────────────────────────┐
│  interpreter.py          │  Optional: LLM generates explanation
│  generate_explanation()  │  using the ACTUAL result rows (no invention)
└────────┬────────────────┘
         │ explanation text + result
         ▼
┌─────────────────────────┐
│  app.py                  │  Renders answer card, chart, table, history
│  (Frontend)              │
└─────────────────────────┘

Supporting modules:
  database.py      — Schema, ingestion, parameterised queries, health check
  llm_providers.py — OpenAI / Gemini / Anthropic adapters (common interface)
  config.py        — Environment-based configuration, never leaks secrets
```

### How a question becomes an answer

1. **Deterministic matching** — `interpreter.py` uses regex patterns to match ~15 common question types without an LLM call. This covers all five required analytics questions and many variations.
2. **LLM fallback** — If no pattern matches, the LLM receives a system prompt describing the allowed operations and returns a structured JSON plan.
3. **Plan validation** — The plan is checked against an allowlist of 12 approved operations, valid field names, and data-type constraints. Invalid plans are rejected before reaching the database.
4. **Deterministic execution** — `analytics.py` maps the operation to a pre-reviewed parameterised SQL query.
5. **Result explanation** — The LLM (or a built-in fallback) generates a plain-English summary using the actual query results.

---

## Technology Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit 1.33+ |
| Data processing | Pandas 2.2+ |
| Database | SQLite (via stdlib `sqlite3`) |
| LLM: Gemini | `google-generativeai` |
| LLM: OpenAI | `openai` v1+ |
| LLM: Anthropic | `anthropic` v0.20+ |
| Configuration | `python-dotenv` |
| Testing | `pytest` 8+ |
| Python | 3.10+ |

---

## Database Schema

```sql
CREATE TABLE transactions (
    Transaction_Id  TEXT    PRIMARY KEY,
    Order_date      TEXT    NOT NULL,     -- ISO-8601: YYYY-MM-DD
    Product         TEXT    NOT NULL,
    Category        TEXT    NOT NULL,
    Region          TEXT    NOT NULL,
    Sales_channel   TEXT    NOT NULL,
    Quantity        INTEGER NOT NULL DEFAULT 0,
    Unit_price      REAL    NOT NULL,
    Discount_amount REAL    NOT NULL DEFAULT 0,
    Status          TEXT    NOT NULL,
    Revenue         REAL    NOT NULL      -- pre-calculated on ingest
);

-- Indexes on frequently filtered/grouped columns
CREATE INDEX idx_txn_date     ON transactions(Order_date);
CREATE INDEX idx_txn_product  ON transactions(Product);
CREATE INDEX idx_txn_region   ON transactions(Region);
CREATE INDEX idx_txn_status   ON transactions(Status);
CREATE INDEX idx_txn_category ON transactions(Category);
```

---

## CSV Ingestion and Data Rules

The ingestion pipeline (`database.py: ingest_csv`) applies the following rules in order:

| Rule | Action |
|---|---|
| Missing required columns | Raise `ValueError` — entire import aborted |
| `Unit_price` is blank | Imputed using per-product median; logged as INFO warning |
| `Unit_price` cannot be imputed (no other price for product) | Row REJECTED and logged |
| `Order_date` is unparseable | Row REJECTED and logged |
| Any required text field is empty | Row REJECTED and logged |
| `Quantity = 0` | Row ACCEPTED with `Revenue = 0`; logged as INFO |
| `Status = Returned` | Row ACCEPTED and stored; EXCLUDED from revenue aggregates |
| Duplicate `Transaction_Id` within the CSV | Second occurrence SKIPPED (first-wins) |
| Re-importing the same CSV (`INSERT OR IGNORE`) | Existing rows silently skipped — **idempotent** |

After import, the report shows `accepted`, `skipped`, and `rejected` counts.

---

## Revenue Calculation Rules

```
Revenue = (Quantity × Unit_price) − Discount_amount
```

**Applied on ingestion** (not at query time) and stored in the `Revenue` column.

| Scenario | Rule |
|---|---|
| Discount is per-transaction (flat amount) | Subtracted once per row |
| `Status = Completed` | Included in all revenue aggregates |
| `Status = Returned` | Stored but excluded from revenue aggregates (filter: `WHERE Status = 'Completed'`) |
| `Quantity = 0, Status = Completed` | Revenue = 0; included (not rejected) |
| Missing `Unit_price` (TXN0051 in sample data) | Imputed via product median before Revenue is calculated |

> **Why exclude Returned?** Returned orders represent revenue reversal. Including them would overstate sales performance. This is documented business policy; it can be changed in `analytics.py` by using `status = "all"` in the plan.

---

## Supported Analytical Questions

All five required questions are supported deterministically (no LLM needed):

| # | Question | Operation |
|---|---|---|
| 1 | What is the total revenue? | `total_revenue` |
| 2 | Which product generated the highest revenue? | `top_products` |
| 3 | What is the monthly revenue trend? | `monthly_revenue` |
| 4 | Which region generated the highest sales? | `top_regions` |
| 5 | How did revenue change between January and February 2026? | `revenue_comparison` |

### Also supported (deterministic)

- Show revenue for January 2026
- Compare January and February 2026
- Which product had the lowest revenue?
- Show revenue by product category
- Compare online and retail sales
- Show the top 5 products by revenue
- How many completed transactions exist?

### Supported via LLM interpretation

Any phrasing of the above, plus questions involving specific product/region filters.

### NOT supported

- Free-form SQL
- Questions about external data sources
- Forecasting or trend prediction
- Customer-level or order-line analytics (not in dataset)

---

## LLM Provider Configuration

### Setting up via `.env`

```bash
cp .env.example .env
# Edit .env with your provider details
```

```env
LLM_PROVIDER=gemini          # openai | gemini | anthropic | groq
LLM_API_KEY=your-key-here
LLM_MODEL=gemini-1.5-flash   # see table below
LLM_TIMEOUT_SECONDS=30
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.0
```

### Recommended models per provider

| Provider | Recommended model | Notes |
|---|---|---|
| `gemini` | `gemini-1.5-flash` | Fast, cost-effective |
| `openai` | `gpt-4o-mini` | Good balance of speed/quality |
| `anthropic` | `claude-3-haiku-20240307` | Fast Haiku tier |
| `groq` | `llama-3.3-70b-versatile` | Ultra-low latency, Llama 3.3 |

### Without an API key

The application works for all 15+ deterministically matched question patterns. Only ambiguous or novel questions require the LLM.

---

## API / Analytical Operations

The core API is internal (Streamlit calls `interpret_question` and `execute_plan` directly). The operations and their plan schemas are:

### `total_revenue`
```json
{"operation": "total_revenue", "filters": {"year": 2026, "month": 1}}
```

### `top_products`
```json
{"operation": "top_products", "filters": {}, "sort_dir": "desc", "limit": 5}
```

### `revenue_comparison`
```json
{"operation": "revenue_comparison", "filters": {"month1": 1, "month2": 2, "year": 2026}}
```

### `monthly_revenue`
```json
{"operation": "monthly_revenue", "filters": {"year": 2026}}
```

### `revenue_by_region`
```json
{"operation": "revenue_by_region", "filters": {}, "sort_dir": "desc", "limit": 10}
```

---

## Installation

### Prerequisites

- Python 3.10 or later
- `pip`

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment (optional — LLM features only)
cp .env.example .env
# Edit .env with your LLM_PROVIDER and LLM_API_KEY

# 4. Load the database
python -X utf8 setup_db.py

# 5. Launch the app
streamlit run app.py
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `gemini` | Provider: `openai`, `gemini`, or `anthropic` |
| `LLM_API_KEY` | _(empty)_ | Your provider API key |
| `LLM_MODEL` | Provider default | Model name (see table above) |
| `LLM_API_BASE_URL` | _(empty)_ | Custom endpoint (e.g. for proxies) |
| `LLM_TIMEOUT_SECONDS` | `30` | Request timeout in seconds |
| `LLM_MAX_TOKENS` | `1024` | Maximum output tokens |
| `LLM_TEMPERATURE` | `0.0` | Sampling temperature (0 = deterministic) |
| `DATABASE_PATH` | `sales.db` | Path to SQLite database file |
| `CSV_PATH` | `sales_data.csv` | Path to source CSV |

---

## Running the Application

```bash
# Initialise/reload database
python -X utf8 setup_db.py

# Start the Streamlit app
streamlit run app.py
```

The UI opens at **http://localhost:8501** by default.

---

## Running Automated Tests

```bash
# Run all 94 tests
python -X utf8 -m pytest tests/ -v

# Run a specific test file
python -X utf8 -m pytest tests/test_analytics.py -v

# Run with coverage (requires pytest-cov)
python -X utf8 -m pytest tests/ --cov=. --cov-report=term-missing
```

### Test suite summary (94 tests)

| File | Tests | Coverage |
|---|---|---|
| `test_agent.py` | 5 | Legacy validator (backwards compat) |
| `test_analytics.py` | 30 | All 5 required analytics + edge cases |
| `test_database.py` | 18 | Ingestion, schema, idempotency, validation |
| `test_interpreter.py` | 29 | Deterministic matching, LLM mocks, plan validation |
| `test_security.py` | 12 | Destructive op blocking, SQL injection, read-only DB |

---

## Security Considerations

| Threat | Mitigation |
|---|---|
| SQL injection | All user input passed as parameterised query values, never string-interpolated |
| Destructive SQL (DROP, DELETE, etc.) | Operation allowlist in `analytics.py`; no SQL is generated by the LLM |
| LLM hallucination of data | LLM only produces a plan; numbers come exclusively from the database |
| API key exposure | Key never logged, never returned in responses, never sent to the browser |
| Write operations | SQLite connection opened in `?mode=ro` (read-only) for all queries |
| Excessive LLM cost | Deterministic matcher handles ~90% of questions without an LLM call |

---

## Performance Notes

Measured on the 51-row sample dataset (representative of the code path):

| Operation | Typical latency |
|---|---|
| Deterministic question match | < 1 ms |
| Database query (indexed) | < 5 ms |
| LLM interpretation (Gemini Flash) | 500–1500 ms |
| LLM explanation generation | 500–1500 ms |
| End-to-end (deterministic path) | < 50 ms |
| End-to-end (LLM path) | 1–3 s |

Indexes on `Order_date`, `Product`, `Region`, `Status`, and `Category` are created during schema initialisation.

Database connections are opened per-request and closed immediately (context manager). For larger datasets, connection pooling via `SQLAlchemy` would be appropriate.

---

## Known Limitations

1. **Dataset size** — The sample CSV contains 51 rows. At >100k rows, aggregation queries would require additional indexes and possibly a migration to PostgreSQL.
2. **No authentication** — This is a single-user local tool. No login or RBAC is implemented.
3. **Single CSV schema** — Ingestion expects the specific column set in `sales_data.csv`. Different CSV schemas require code changes.
4. **LLM quality varies** — The LLM path depends on the provider and model. Weaker models may return invalid plans (which are caught and reported to the user).
5. **API key stored in session** — The key entered via the UI sidebar is held in Streamlit session state (in-memory only). It is not persisted to disk. Refreshing the page clears it.
6. **No real-time data** — The database must be manually reloaded after CSV updates.
7. **Anthropic/OpenAI not live-tested** — Only Gemini is available for live testing. OpenAI and Anthropic adapters are implemented and unit-tested with mocks. See `llm_providers.py`.

---

## Future Improvements

- Connection pooling for high-concurrency deployments
- CSV upload through the UI (drag-and-drop)
- Semantic caching of LLM interpretations (same question → same plan, no second call)
- Self-correcting LLM loop (feed DB errors back for plan refinement)
- Export results to CSV/Excel
- Role-based access control for multi-user deployments
- PostgreSQL migration for production-scale data
- Time-series forecasting using the stored monthly trends
