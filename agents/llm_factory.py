"""
FinRegAgents – LLM Factory

Unterstützte Provider:
  anthropic  – Claude Sonnet / Opus / Haiku  (Standard)
  openai     – GPT-4o, GPT-4o-mini, o3
  gemini     – Gemini 3.1 Pro, 2.5 Pro
  mistral    – Mistral Large, Mistral Small
  cohere     – Command R+, Command A
  grok       – Grok-3, Grok-3 Mini  (OpenAI-kompatibler Endpunkt)
  ollama     – Llama 3.3, Qwen2.5 etc. (lokal, kein API-Key)

Verwendung:
    from agents.llm_factory import build_llm, PROVIDER_DEFAULTS

    llm = build_llm("gemini", model="gemini-3.1-pro-001", temperature=0.1)
    llm = build_llm("ollama", model="llama3.3")
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Provider-Defaults
# ------------------------------------------------------------------ #

PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "anthropic": {
        "model": "claude-sonnet-4-6",
        "max_tokens": 2048,
    },
    "openai": {
        "model": "gpt-4o",
        "max_tokens": 2048,
    },
    "gemini": {
        "model": "gemini-3.1-pro-001",
        "max_tokens": 2048,
    },
    "mistral": {
        "model": "mistral-large-latest",
        "max_tokens": 2048,
    },
    "cohere": {
        "model": "command-r-plus",
        "max_tokens": 2048,
    },
    "grok": {
        "model": "grok-3",
        "max_tokens": 2048,
    },
    "ollama": {
        "model": "llama3.3",
        # Ollama hat kein festes Token-Limit – wird über num_predict gesetzt
    },
}

VALID_PROVIDERS = set(PROVIDER_DEFAULTS.keys())

# Mindest-Empfehlung für regulatorische Qualität bei Ollama
OLLAMA_MIN_RECOMMENDED_MODELS = {
    "llama3.3",
    "llama3.3:70b",
    "qwen2.5:72b",
    "qwen2.5:32b",
    "mistral",
    "mixtral",
}


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #


def build_llm(
    provider: str,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int | None = None,
    **kwargs: Any,
):
    """
    Gibt eine LangChain-kompatible BaseChatModel-Instanz zurück.

    Parameter
    ---------
    provider : str
        Einer von: anthropic, openai, gemini, mistral, cohere, grok, ollama
    model : str | None
        Modellname. Wenn None, wird der Provider-Default verwendet.
    temperature : float
        Sampling-Temperatur (0.0–1.0). Default 0.1 für deterministische Prüfungen.
    max_tokens : int | None
        Maximale Ausgabe-Tokens. None = Provider-Default.
    **kwargs
        Weitere Provider-spezifische Parameter (z.B. base_url für Ollama).
    """
    provider = provider.lower().strip()
    if provider not in VALID_PROVIDERS:
        raise ValueError(
            f"Unbekannter Provider: '{provider}'. "
            f"Gültige Provider: {sorted(VALID_PROVIDERS)}"
        )

    defaults = PROVIDER_DEFAULTS[provider]
    resolved_model = model or defaults["model"]
    resolved_max_tokens = max_tokens or defaults.get("max_tokens", 2048)

    logger.info("LLM-Factory: provider=%s model=%s", provider, resolved_model)

    if provider == "anthropic":
        return _build_anthropic(
            resolved_model, temperature, resolved_max_tokens, **kwargs
        )
    if provider == "openai":
        return _build_openai(resolved_model, temperature, resolved_max_tokens, **kwargs)
    if provider == "gemini":
        return _build_gemini(resolved_model, temperature, resolved_max_tokens, **kwargs)
    if provider == "mistral":
        return _build_mistral(
            resolved_model, temperature, resolved_max_tokens, **kwargs
        )
    if provider == "cohere":
        return _build_cohere(resolved_model, temperature, resolved_max_tokens, **kwargs)
    if provider == "grok":
        return _build_grok(resolved_model, temperature, resolved_max_tokens, **kwargs)
    if provider == "ollama":
        return _build_ollama(resolved_model, temperature, resolved_max_tokens, **kwargs)


# ------------------------------------------------------------------ #
# Provider-Implementierungen
# ------------------------------------------------------------------ #


def _build_anthropic(model, temperature, max_tokens, **kwargs):
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError(
            "langchain-anthropic ist nicht installiert. "
            "Installieren: pip install langchain-anthropic"
        )
    _require_env("ANTHROPIC_API_KEY", "anthropic")
    return ChatAnthropic(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def _build_openai(model, temperature, max_tokens, **kwargs):
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai ist nicht installiert. "
            "Installieren: pip install langchain-openai"
        )
    _require_env("OPENAI_API_KEY", "openai")
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def _build_gemini(model, temperature, max_tokens, **kwargs):
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError:
        raise ImportError(
            "langchain-google-genai ist nicht installiert. "
            "Installieren: pip install langchain-google-genai"
        )
    _require_env("GOOGLE_API_KEY", "gemini")
    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        max_output_tokens=max_tokens,
        **kwargs,
    )


def _build_mistral(model, temperature, max_tokens, **kwargs):
    try:
        from langchain_mistralai import ChatMistralAI
    except ImportError:
        raise ImportError(
            "langchain-mistralai ist nicht installiert. "
            "Installieren: pip install langchain-mistralai"
        )
    _require_env("MISTRAL_API_KEY", "mistral")
    return ChatMistralAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def _build_cohere(model, temperature, max_tokens, **kwargs):
    try:
        from langchain_cohere import ChatCohere
    except ImportError:
        raise ImportError(
            "langchain-cohere ist nicht installiert. "
            "Installieren: pip install langchain-cohere"
        )
    _require_env("COHERE_API_KEY", "cohere")
    return ChatCohere(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )


def _build_grok(model, temperature, max_tokens, **kwargs):
    """
    xAI Grok über den OpenAI-kompatiblen Endpunkt.
    Benötigt: XAI_API_KEY + base_url https://api.x.ai/v1
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError(
            "langchain-openai ist nicht installiert. "
            "Installieren: pip install langchain-openai"
        )
    _require_env("XAI_API_KEY", "grok")
    base_url = kwargs.pop("base_url", "https://api.x.ai/v1")
    api_key = os.environ["XAI_API_KEY"]
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )


def _build_ollama(model, temperature, max_tokens, **kwargs):
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        raise ImportError(
            "langchain-ollama ist nicht installiert. "
            "Installieren: pip install langchain-ollama"
        )
    base_url = kwargs.pop(
        "base_url", os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )

    if model not in OLLAMA_MIN_RECOMMENDED_MODELS:
        logger.warning(
            "Ollama-Modell '%s' ist nicht in der empfohlenen Modell-Liste für "
            "regulatorische Prüfqualität. Empfohlen: %s",
            model,
            sorted(OLLAMA_MIN_RECOMMENDED_MODELS),
        )

    return ChatOllama(
        model=model,
        temperature=temperature,
        num_predict=max_tokens,
        base_url=base_url,
        **kwargs,
    )


# ------------------------------------------------------------------ #
# Hilfsfunktionen
# ------------------------------------------------------------------ #


def _require_env(var: str, provider: str) -> None:
    if var == "GOOGLE_API_KEY" and not os.environ.get(var):
        gemini_key = os.environ.get("GEMINI_API_KEY")
        if gemini_key:
            os.environ[var] = gemini_key
            logger.info("GEMINI_API_KEY erkannt – als GOOGLE_API_KEY übernommen.")
    if not os.environ.get(var):
        raise EnvironmentError(
            f"Umgebungsvariable '{var}' fehlt für Provider '{provider}'. "
            f"Bitte in .env oder Shell setzen."
        )


def list_providers() -> list[str]:
    """Gibt alle unterstützten Provider-Namen zurück."""
    return sorted(VALID_PROVIDERS)


def default_model(provider: str) -> str:
    """Gibt das Default-Modell für einen Provider zurück."""
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unbekannter Provider: '{provider}'")
    return PROVIDER_DEFAULTS[provider]["model"]
