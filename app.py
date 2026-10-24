"""
app.py – AI Business Intelligence Assistant (Streamlit frontend).

Architecture:
  - app.py       : Streamlit UI only; no business logic.
  - interpreter.py: Natural-language → validated analytical plan.
  - analytics.py : Deterministic SQL analytics; all numbers come from here.
  - database.py  : DB connection, schema, ingestion.
  - llm_providers.py: Provider-independent LLM adapter.
  - config.py    : Configuration from environment / session.

The LLM interprets questions and generates explanations.
It NEVER calculates sales numbers.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st

# ── Local modules ────────────────────────────────────────────────────────────
from analytics import execute_plan, get_dashboard_summary
from config import SUPPORTED_PROVIDERS, DEFAULT_MODELS, load_config
from database import database_status, ingest_csv, init_schema
from interpreter import generate_explanation, interpret_question

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="AI BI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .main { background: #0f1117; }

  /* Metric cards */
  .metric-card {
    background: linear-gradient(135deg, #1e2130 0%, #252840 100%);
    border: 1px solid #2d3250;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }
  .metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.2);
  }
  .metric-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9ca3af;
    margin-bottom: 8px;
  }
  .metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
  }
  .metric-sub {
    font-size: 0.7rem;
    color: #6b7280;
    margin-top: 4px;
  }

  /* Status badges */
  .badge-ok   { background:#166534; color:#bbf7d0; padding:3px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; }
  .badge-warn { background:#713f12; color:#fef3c7; padding:3px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; }
  .badge-err  { background:#7f1d1d; color:#fecaca; padding:3px 10px; border-radius:99px; font-size:0.72rem; font-weight:600; }

  /* Answer card */
  .answer-card {
    background: linear-gradient(135deg, #1a1f35 0%, #1e2540 100%);
    border: 1px solid #3b4270;
    border-left: 4px solid #6366f1;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 16px 0;
  }

  /* Suggested question chips */
  .chip-row { display: flex; flex-wrap: wrap; gap: 8px; margin: 8px 0; }

  /* Section headers */
  .section-header {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6366f1;
    margin: 20px 0 8px 0;
  }

  /* Divider */
  hr.thin { border: none; border-top: 1px solid #2d3250; margin: 12px 0; }
</style>
""", unsafe_allow_html=True)

# ── Session state defaults ───────────────────────────────────────────────────
def _init_state() -> None:
    current_prov = os.getenv("LLM_PROVIDER", "gemini")
    prov_key = (
        os.getenv("GROQ_API_KEY") if current_prov == "groq"
        else os.getenv("OPENAI_API_KEY") if current_prov == "openai"
        else os.getenv("ANTHROPIC_API_KEY") if current_prov == "anthropic"
        else os.getenv("GEMINI_API_KEY") if current_prov == "gemini"
        else os.getenv("LLM_API_KEY", "")
    ) or os.getenv("LLM_API_KEY", "")

    defaults = {
        "history":         [],          # list of {question, result, explanation, path, elapsed}
        "provider":        current_prov,
        "api_key":         prov_key,
        "model":           os.getenv("LLM_MODEL", "") or DEFAULT_MODELS.get(current_prov, ""),
        "last_question":   "",
        "ingestion_report": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

DB_PATH  = os.getenv("DATABASE_PATH", "sales.db")
CSV_PATH = os.getenv("CSV_PATH", "sales_data.csv")

# ── Helper: get LLM config from session ─────────────────────────────────────
def _llm_config():
    try:
        return load_config(
            provider=st.session_state.provider,
            api_key=st.session_state.api_key,
            model=st.session_state.model or None,
        )
    except Exception:
        return None

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("<div class='section-header'>LLM Provider</div>", unsafe_allow_html=True)

    def _on_provider_change():
        p = st.session_state.provider
        st.session_state.model = DEFAULT_MODELS.get(p, "")
        # Automatically pull provider-specific env var if current key is empty
        prov_env = {
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
        }.get(p, "")
        if not st.session_state.api_key and prov_env and os.getenv(prov_env):
            st.session_state.api_key = os.getenv(prov_env)

    # Provider selector
    provider_idx = list(SUPPORTED_PROVIDERS).index(st.session_state.provider) \
        if st.session_state.provider in SUPPORTED_PROVIDERS else 0
    chosen_provider = st.selectbox(
        "Provider", SUPPORTED_PROVIDERS, index=provider_idx, key="provider",
        on_change=_on_provider_change,
        help="Select the AI provider. Requires an API key.",
    )

    # Model field (pre-filled with default for provider)
    default_model = DEFAULT_MODELS.get(chosen_provider, "")
    if not st.session_state.model:
        st.session_state.model = default_model

    st.text_input(
        "Model", key="model",
        help=f"Default for {chosen_provider}: {default_model}",
    )

    # API key entry
    st.text_input(
        "API Key", type="password", key="api_key",
        help="Your provider API key. Not stored permanently. Never sent to the browser.",
    )

    # Connection status
    cfg = _llm_config()
    if cfg and cfg.llm.is_valid():
        st.markdown('<span class="badge-ok">🟢 LLM Configured</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-warn">🟡 LLM Not Configured</span>', unsafe_allow_html=True)
        st.caption("Enter an API key to enable AI interpretation. "
                   "Common questions still work without it.")

    st.markdown("---")
    st.markdown("## 📂 Data Management")

    db_stat = database_status(DB_PATH)
    if db_stat["connected"]:
        st.markdown(
            f'<span class="badge-ok">🟢 DB Connected – '
            f'{db_stat["total_transactions"]} rows</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<span class="badge-err">🔴 DB Error: {db_stat.get("error","")}</span>',
            unsafe_allow_html=True,
        )

    if st.button("🔄 (Re)Import CSV", help="Load or reload data from sales_data.csv"):
        if not Path(CSV_PATH).exists():
            st.error(f"CSV not found: {CSV_PATH}")
        else:
            with st.spinner("Importing…"):
                try:
                    report = ingest_csv(CSV_PATH, DB_PATH)
                    st.session_state.ingestion_report = report
                    st.success(report.summary())
                except Exception as exc:
                    st.error(f"Import failed: {exc}")

    if st.session_state.ingestion_report:
        rpt = st.session_state.ingestion_report
        if rpt.warnings:
            with st.expander("⚠️ Ingestion Warnings", expanded=False):
                for w in rpt.warnings:
                    st.caption(w)

    st.markdown("---")
    st.caption("AI BI Assistant v2.0 | Data stays local | Keys never logged")


# ── Ensure DB exists ─────────────────────────────────────────────────────────
if not Path(DB_PATH).exists():
    try:
        init_schema(DB_PATH)
        ingest_csv(CSV_PATH, DB_PATH)
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<h1 style='font-size:2rem; font-weight:700; margin-bottom:4px;'>
  📊 AI Business Intelligence Assistant
</h1>
<p style='color:#9ca3af; font-size:0.95rem; margin-bottom:24px;'>
  Ask natural-language questions about your sales data. Powered by deterministic analytics + AI interpretation.
</p>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# DASHBOARD SUMMARY CARDS
# ═══════════════════════════════════════════════════════════════════════════════
summary = get_dashboard_summary(DB_PATH)

def _fmt_currency(val) -> str:
    if val is None:
        return "N/A"
    return f"£{val:,.0f}"

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Total Revenue</div>
      <div class="metric-value">{_fmt_currency(summary['total_revenue'])}</div>
      <div class="metric-sub">Completed transactions only</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Transactions</div>
      <div class="metric-value">{summary['total_transactions'] or 'N/A'}</div>
      <div class="metric-sub">Completed orders</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Top Product</div>
      <div class="metric-value" style="font-size:1.1rem">{summary['top_product']}</div>
      <div class="metric-sub">By revenue</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">Best Region</div>
      <div class="metric-value" style="font-size:1.2rem">{summary['top_region']}</div>
      <div class="metric-sub">By revenue</div>
    </div>""", unsafe_allow_html=True)

if summary["date_from"] and summary["date_from"] != "N/A":
    st.caption(f"📅 Reporting period: {summary['date_from']} → {summary['date_to']}")

st.markdown("<hr class='thin'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SUGGESTED QUESTIONS
# ═══════════════════════════════════════════════════════════════════════════════
SUGGESTED_QUESTIONS = [
    "What is the total revenue?",
    "Which product generated the highest revenue?",
    "Show the monthly revenue trend",
    "Which region generated the highest sales?",
    "Compare revenue between January and February 2026",
    "Which category performed best?",
    "Show the top 5 products by revenue",
    "Compare online and retail sales",
    "How many completed transactions exist?",
]

st.markdown("<div class='section-header'>💡 Suggested Questions</div>", unsafe_allow_html=True)

# Display as clickable buttons in a flex row
cols = st.columns(3)
for i, sq in enumerate(SUGGESTED_QUESTIONS):
    with cols[i % 3]:
        if st.button(sq, key=f"sq_{i}", use_container_width=True):
            st.session_state.last_question = sq


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY INPUT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr class='thin'>", unsafe_allow_html=True)
st.markdown("<div class='section-header'>🔍 Ask a Question</div>", unsafe_allow_html=True)

col_input, col_btn = st.columns([5, 1])
with col_input:
    question = st.text_input(
        "Natural-language question",
        value=st.session_state.last_question,
        placeholder="e.g. Which product generated the highest revenue in March 2026?",
        label_visibility="collapsed",
        key="question_input",
        max_chars=1000,
    )
with col_btn:
    ask_clicked = st.button("Ask ➤", type="primary", use_container_width=True)

# Clear button
if st.button("✕ Clear", key="clear_btn"):
    st.session_state.last_question = ""
    st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# QUESTION PROCESSING
# ═══════════════════════════════════════════════════════════════════════════════
def _process_question(q: str) -> None:
    q = q.strip()
    if not q:
        st.warning("Please enter a question.")
        return
    if len(q) > 1000:
        st.error("Question too long (max 1 000 characters).")
        return
    if not database_status(DB_PATH).get("connected"):
        st.error("Database is not available. Please import the CSV first using the sidebar.")
        return

    cfg = _llm_config()
    llm_cfg = cfg.llm if cfg else None

    t0 = time.perf_counter()

    with st.spinner("Interpreting your question…"):
        plan, interp_msg, path = interpret_question(q, llm_cfg)

    if plan is None:
        st.error(f"**Could not process question:** {interp_msg}")
        if path == "unsupported":
            st.info(
                "This question type is not currently supported. "
                "Try one of the suggested questions above."
            )
        return

    with st.spinner("Running analytics…"):
        try:
            result = execute_plan(plan, DB_PATH)
        except Exception as exc:
            st.error(f"**Analytics error:** {exc}")
            logger.error("Analytics execution failed: %s", exc, exc_info=True)
            return

    # Generate explanation
    with st.spinner("Generating explanation…"):
        explanation = generate_explanation(q, result, llm_cfg)

    elapsed = time.perf_counter() - t0

    # Save to history
    st.session_state.history.insert(0, {
        "question":    q,
        "result":      result,
        "explanation": explanation,
        "path":        path,
        "elapsed":     elapsed,
        "plan":        plan,
    })
    # Keep last 20 entries
    st.session_state.history = st.session_state.history[:20]


if ask_clicked and question:
    _process_question(question)
elif ask_clicked:
    st.warning("Please enter a question.")


# ═══════════════════════════════════════════════════════════════════════════════
# ANSWER PANEL
# ═══════════════════════════════════════════════════════════════════════════════
if st.session_state.history:
    latest = st.session_state.history[0]
    result      = latest["result"]
    explanation = latest["explanation"]
    plan        = latest["plan"]
    path        = latest["path"]
    elapsed     = latest["elapsed"]

    st.markdown("<div class='section-header'>📋 Answer</div>", unsafe_allow_html=True)

    # Answer card
    path_badge = {
        "deterministic": "⚡ Deterministic",
        "llm":           "🤖 AI-interpreted",
    }.get(path, path)

    st.markdown(f"""
    <div class="answer-card">
      <div style="font-size:0.75rem; color:#6366f1; font-weight:600; margin-bottom:8px;">
        {plan['operation'].upper().replace('_', ' ')} &nbsp;·&nbsp; {path_badge}
        &nbsp;·&nbsp; {elapsed:.2f}s
      </div>
      <div style="font-size:1.05rem; line-height:1.6; color:#e2e8f0;">
        {explanation}
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Results table and chart
    rows = result.get("rows", [])
    cols_out = result.get("columns", [])
    op = result.get("operation", "")

    if not rows:
        st.info("No data matches the specified criteria.")
    else:
        df = pd.DataFrame(rows, columns=cols_out)

        # ── Currency formatting in display ─────────────────────────────────
        def _format_df(df_: pd.DataFrame) -> pd.DataFrame:
            out = df_.copy()
            for col_ in out.columns:
                if out[col_].dtype in ("float64", "float32") or "revenue" in col_.lower():
                    try:
                        out[col_] = out[col_].apply(
                            lambda v: f"£{v:,.2f}" if pd.notna(v) else "N/A"
                        )
                    except Exception:
                        pass
            return out

        def _safe_dataframe(df_to_show: pd.DataFrame) -> None:
            try:
                st.dataframe(df_to_show, use_container_width=True, hide_index=True)
            except Exception:
                st.table(df_to_show)

        # ── Choose visualisation ───────────────────────────────────────────
        if op == "monthly_revenue" and len(df) > 0:
            col_t, col_ch = st.columns([1, 2])
            with col_t:
                _safe_dataframe(_format_df(df))
            with col_ch:
                chart_df = df.copy()
                chart_df.columns = ["Month", "Revenue"]
                chart_df["Revenue"] = pd.to_numeric(chart_df["Revenue"], errors="coerce")
                st.bar_chart(chart_df.set_index("Month"), use_container_width=True)

        elif op in ("revenue_by_product", "revenue_by_region",
                    "revenue_by_category", "revenue_by_channel",
                    "top_products", "top_regions") and len(df) > 0:
            col_t, col_ch = st.columns([1, 2])
            with col_t:
                _safe_dataframe(_format_df(df))
            with col_ch:
                chart_df = df.copy()
                chart_df.columns = [chart_df.columns[0], "Revenue"]
                chart_df["Revenue"] = pd.to_numeric(chart_df["Revenue"], errors="coerce")
                st.bar_chart(
                    chart_df.set_index(chart_df.columns[0]),
                    use_container_width=True,
                )

        elif op == "revenue_comparison" and len(df) >= 2:
            col1, col2 = st.columns(2)
            for i, row in enumerate(rows):
                with (col1 if i == 0 else col2):
                    val = row[1] or 0
                    st.markdown(f"""
                    <div class="metric-card">
                      <div class="metric-label">{row[0]}</div>
                      <div class="metric-value">{_fmt_currency(val)}</div>
                    </div>""", unsafe_allow_html=True)
            st.markdown("")
            chart_df = pd.DataFrame(rows, columns=["Period", "Revenue"])
            chart_df["Revenue"] = pd.to_numeric(chart_df["Revenue"], errors="coerce")
            st.bar_chart(chart_df.set_index("Period"), use_container_width=True)

        else:
            _safe_dataframe(_format_df(df))

        # ── Technical details expander ─────────────────────────────────────
        with st.expander("🔧 Technical Details", expanded=False):
            st.json({
                "operation":    op,
                "plan":         plan,
                "rows_returned": len(rows),
                "path_used":    path,
                "elapsed_s":    round(elapsed, 3),
                "meta":         result.get("meta", {}),
            })

    st.markdown("<hr class='thin'>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# QUERY HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
if len(st.session_state.history) > 1:
    st.markdown("<div class='section-header'>🕒 Recent Questions</div>", unsafe_allow_html=True)
    for i, item in enumerate(st.session_state.history[1:], start=1):
        col_q, col_rerun = st.columns([5, 1])
        with col_q:
            st.caption(f"**{i}.** {item['question']}  •  {item['plan']['operation']}  •  {item['elapsed']:.2f}s")
        with col_rerun:
            if st.button("↩ Rerun", key=f"rerun_{i}", use_container_width=True):
                _process_question(item["question"])
                st.rerun()
