"""
config.py – Centralised configuration for the AI BI Assistant.

Priority (highest first):
  1. Values passed explicitly at runtime (e.g. from the Streamlit UI session)
  2. Environment variables (from .env or the host environment)
  3. Hard-coded defaults below
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)          # don't override real env vars
except ImportError:
    pass                                 # dotenv is optional


# ---------------------------------------------------------------------------
# Provider constants
# ---------------------------------------------------------------------------
SUPPORTED_PROVIDERS = ("openai", "gemini", "anthropic", "groq")

DEFAULT_MODELS = {
    "openai":    "gpt-4o-mini",
    "gemini":    "gemini-1.5-flash",
    "anthropic": "claude-3-haiku-20240307",
    "groq":      "llama-3.3-70b-versatile",
}


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------
@dataclass
class LLMConfig:
    provider:      str
    api_key:       str
    model:         str
    api_base_url:  Optional[str]
    timeout:       int
    max_tokens:    int
    temperature:   float

    def is_valid(self) -> bool:
        return bool(self.provider and self.api_key and self.model)

    def redacted(self) -> dict:
        """Safe representation – never exposes the key."""
        return {
            "provider":    self.provider,
            "model":       self.model,
            "api_base_url": self.api_base_url or "(default)",
            "timeout":     self.timeout,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "key_set":     bool(self.api_key),
        }


@dataclass
class AppConfig:
    database_path: str
    csv_path:      str
    llm:           LLMConfig


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def load_config(
    *,
    provider:     Optional[str] = None,
    api_key:      Optional[str] = None,
    model:        Optional[str] = None,
    api_base_url: Optional[str] = None,
) -> AppConfig:
    """
    Build an AppConfig from environment variables, with optional overrides.

    Parameters passed explicitly (e.g. from the UI) take precedence over
    the environment.
    """
    resolved_provider = (
        provider
        or os.getenv("LLM_PROVIDER", "gemini")
    ).lower().strip()

    if resolved_provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM provider '{resolved_provider}'. "
            f"Choose one of: {', '.join(SUPPORTED_PROVIDERS)}"
        )

    resolved_model = (
        model
        or os.getenv("LLM_MODEL", "")
        or DEFAULT_MODELS[resolved_provider]
    )

    provider_key_env = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
    }.get(resolved_provider, "")

    resolved_key = (
        api_key
        or (os.getenv(provider_key_env) if provider_key_env else "")
        or os.getenv("LLM_API_KEY", "")
    )
    resolved_base = api_base_url or os.getenv("LLM_API_BASE_URL", "") or None

    llm_cfg = LLMConfig(
        provider=resolved_provider,
        api_key=resolved_key,
        model=resolved_model,
        api_base_url=resolved_base,
        timeout=int(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1024")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.0")),
    )

    return AppConfig(
        database_path=os.getenv("DATABASE_PATH", "sales.db"),
        csv_path=os.getenv("CSV_PATH", "sales_data.csv"),
        llm=llm_cfg,
    )
