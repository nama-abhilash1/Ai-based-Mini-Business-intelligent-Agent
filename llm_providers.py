"""
llm_providers.py – Provider-independent LLM abstraction.

Hierarchy:
  BaseLLMProvider (abstract)
  ├── OpenAIProvider
  ├── GeminiProvider
  └── AnthropicProvider

Each provider:
  - Accepts a structured prompt string.
  - Returns a plain-text response string.
  - Handles timeouts, rate limits, invalid credentials, and malformed responses.
  - Never exposes the API key in logs or return values.
  - Raises LLMError for all provider-specific errors (normalized).

Usage:
    from llm_providers import get_provider
    from config import load_config

    cfg = load_config()
    provider = get_provider(cfg.llm)
    text = provider.complete(system_prompt, user_prompt)
"""
from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Optional

from config import LLMConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Normalised error
# ---------------------------------------------------------------------------
class LLMError(Exception):
    """Raised for any provider-level failure. Never contains the API key."""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------
class BaseLLMProvider(ABC):
    def __init__(self, cfg: LLMConfig):
        self._cfg = cfg

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Send a prompt and return the model's text response."""

    def _safe_log(self, msg: str) -> None:
        """Log without leaking the API key."""
        safe = msg.replace(self._cfg.api_key, "***") if self._cfg.api_key else msg
        logger.debug(safe)


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------
class OpenAIProvider(BaseLLMProvider):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import openai
        except ImportError:
            raise LLMError(
                "The 'openai' package is not installed. Run: pip install openai"
            )

        kwargs: dict = {
            "model":       self._cfg.model,
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_prompt},
            ],
            "temperature": self._cfg.temperature,
            "max_tokens":  self._cfg.max_tokens,
            "timeout":     self._cfg.timeout,
        }

        try:
            base = self._cfg.api_base_url
            client = openai.OpenAI(
                api_key  = self._cfg.api_key,
                base_url = base if base else None,
            )
            response = client.chat.completions.create(**kwargs)
            return response.choices[0].message.content or ""
        except openai.AuthenticationError:
            raise LLMError("OpenAI: Invalid API key. Check LLM_API_KEY.", retryable=False)
        except openai.RateLimitError:
            raise LLMError("OpenAI: Rate limit exceeded. Try again later.", retryable=True)
        except openai.APITimeoutError:
            raise LLMError("OpenAI: Request timed out.", retryable=True)
        except openai.APIConnectionError as e:
            raise LLMError(f"OpenAI: Connection error – {type(e).__name__}.", retryable=True)
        except openai.BadRequestError as e:
            raise LLMError(f"OpenAI: Bad request – {e.message}.", retryable=False)
        except Exception as e:
            # Do NOT include e's str() directly – it might contain the key in some SDKs
            raise LLMError(f"OpenAI: Unexpected error – {type(e).__name__}.", retryable=False)


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------
class GeminiProvider(BaseLLMProvider):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import google.generativeai as genai
        except ImportError:
            raise LLMError(
                "The 'google-generativeai' package is not installed. "
                "Run: pip install google-generativeai"
            )

        try:
            genai.configure(api_key=self._cfg.api_key)
            try:
                model = genai.GenerativeModel(
                    model_name         = self._cfg.model,
                    system_instruction = system_prompt,
                )
                prompt_to_send = user_prompt
            except TypeError:
                model = genai.GenerativeModel(model_name=self._cfg.model)
                prompt_to_send = f"System Instructions:\n{system_prompt}\n\nUser Question:\n{user_prompt}"

            gen_config = genai.GenerationConfig(
                temperature       = self._cfg.temperature,
                max_output_tokens = self._cfg.max_tokens,
            )

            try:
                response = model.generate_content(
                    prompt_to_send,
                    generation_config = gen_config,
                    request_options   = {"timeout": self._cfg.timeout},
                )
            except TypeError:
                response = model.generate_content(
                    prompt_to_send,
                    generation_config = gen_config,
                )

            # response.text raises if blocked
            return response.text or ""
        except Exception as e:
            err_str = str(e)
            # Classify common error patterns without leaking the key
            if "API_KEY_INVALID" in err_str or "API key not valid" in err_str:
                raise LLMError("Gemini: Invalid API key. Check LLM_API_KEY.", retryable=False)
            if "QUOTA_EXCEEDED" in err_str or "quota" in err_str.lower():
                raise LLMError("Gemini: Quota exceeded. Try again later.", retryable=True)
            if "timeout" in err_str.lower() or "deadline" in err_str.lower():
                raise LLMError("Gemini: Request timed out.", retryable=True)
            if "blocked" in err_str.lower() or "SAFETY" in err_str:
                raise LLMError("Gemini: Response was blocked by safety filters.", retryable=False)
            raise LLMError(f"Gemini: {type(e).__name__} – see server logs.", retryable=False)


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------
class AnthropicProvider(BaseLLMProvider):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        try:
            import anthropic
        except ImportError:
            raise LLMError(
                "The 'anthropic' package is not installed. Run: pip install anthropic"
            )

        try:
            client = anthropic.Anthropic(
                api_key = self._cfg.api_key,
            )
            message = client.messages.create(
                model      = self._cfg.model,
                max_tokens = self._cfg.max_tokens,
                system     = system_prompt,
                messages   = [{"role": "user", "content": user_prompt}],
                timeout    = float(self._cfg.timeout),
            )
            return message.content[0].text if message.content else ""
        except anthropic.AuthenticationError:
            raise LLMError("Anthropic: Invalid API key. Check LLM_API_KEY.", retryable=False)
        except anthropic.RateLimitError:
            raise LLMError("Anthropic: Rate limit exceeded. Try again later.", retryable=True)
        except anthropic.APITimeoutError:
            raise LLMError("Anthropic: Request timed out.", retryable=True)
        except anthropic.APIConnectionError as e:
            raise LLMError(f"Anthropic: Connection error – {type(e).__name__}.", retryable=True)
        except Exception as e:
            raise LLMError(f"Anthropic: Unexpected error – {type(e).__name__}.", retryable=False)


# ---------------------------------------------------------------------------
# Groq provider
# ---------------------------------------------------------------------------
class GroqProvider(BaseLLMProvider):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        client = None
        # Prefer the dedicated groq package, fallback to openai client with Groq base URL
        try:
            from groq import Groq
            client = Groq(
                api_key  = self._cfg.api_key,
                base_url = self._cfg.api_base_url or None,
            )
        except ImportError:
            try:
                import openai
                client = openai.OpenAI(
                    api_key  = self._cfg.api_key,
                    base_url = self._cfg.api_base_url or "https://api.groq.com/openai/v1",
                )
            except ImportError:
                raise LLMError(
                    "Neither 'groq' nor 'openai' package is installed. "
                    "Run: pip install groq"
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        model = self._cfg.model or "llama-3.3-70b-versatile"
        # If the user switched providers and the model field still has a gemini/openai/anthropic model
        # or a deprecated Groq checkpoint, default to Groq's premier production model
        if any(model.startswith(p) for p in ("gemini", "gpt", "claude")):
            model = "llama-3.3-70b-versatile"
        elif model in ("llama3-70b-8192", "llama3-8b-8192", "llama-3.1-70b-versatile"):
            model = "llama-3.3-70b-versatile"

        try:
            response = client.chat.completions.create(
                model       = model,
                messages    = messages,
                temperature = self._cfg.temperature,
                max_tokens  = self._cfg.max_tokens,
                timeout     = float(self._cfg.timeout),
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_name = type(e).__name__
            err_str = str(e).lower()
            if "notfound" in err_name.lower() or "404" in err_str or "not_found" in err_str:
                raise LLMError(
                    f"Groq: Model '{model}' not found. Please choose a valid Groq model like 'llama-3.3-70b-versatile' or 'llama-3.1-8b-instant'.",
                    retryable=False,
                )
            if "auth" in err_name.lower() or "invalid api key" in err_str:
                raise LLMError("Groq: Invalid API key. Check GROQ_API_KEY or LLM_API_KEY.", retryable=False)
            if "ratelimit" in err_name.lower() or "rate limit" in err_str:
                raise LLMError("Groq: Rate limit exceeded. Try again later.", retryable=True)
            if "timeout" in err_name.lower() or "timed out" in err_str:
                raise LLMError("Groq: Request timed out.", retryable=True)
            if "connection" in err_name.lower() or "network" in err_str:
                raise LLMError(f"Groq: Connection error – {err_name}.", retryable=True)
            if "badrequest" in err_name.lower() or "400" in err_str:
                raise LLMError(f"Groq: Bad request – {err_name}.", retryable=False)
            raise LLMError(f"Groq: Unexpected error – {err_name}.", retryable=False)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def get_provider(cfg: LLMConfig) -> BaseLLMProvider:
    """Return the correct provider adapter for the given config."""
    mapping = {
        "openai":    OpenAIProvider,
        "gemini":    GeminiProvider,
        "anthropic": AnthropicProvider,
        "groq":      GroqProvider,
    }
    cls = mapping.get(cfg.provider)
    if cls is None:
        raise LLMError(f"Unknown provider '{cfg.provider}'.", retryable=False)
    return cls(cfg)
