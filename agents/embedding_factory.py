"""
FinRegAgents – Embedding Factory

Unterstützte Embedding-Provider:
  openai    – text-embedding-3-small/large  (Standard, benötigt OPENAI_API_KEY)
  fastembed – BAAI/bge-small-en-v1.5        (lokal, kein API-Key)
  gemini    – text-embedding-004            (benötigt GOOGLE_API_KEY)
  mistral   – mistral-embed                 (benötigt MISTRAL_API_KEY)
  ollama    – nomic-embed-text, mxbai-embed-large (lokal, kein API-Key)

Verwendung:
    from agents.embedding_factory import build_embedding, EMBEDDING_DEFAULTS

    embed = build_embedding("gemini")
    embed = build_embedding("ollama", model="nomic-embed-text")
    embed = build_embedding("fastembed")   # kein Key nötig
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Provider-Defaults
# ------------------------------------------------------------------ #

EMBEDDING_DEFAULTS: dict[str, dict[str, Any]] = {
    "openai": {
        "model": "text-embedding-3-small",
    },
    "fastembed": {
        "model": "BAAI/bge-small-en-v1.5",
    },
    "gemini": {
        "model": "text-embedding-004",
    },
    "mistral": {
        "model": "mistral-embed",
    },
    "ollama": {
        "model": "nomic-embed-text",
    },
}

VALID_EMBEDDING_PROVIDERS = set(EMBEDDING_DEFAULTS.keys())

# Provider, die keinen API-Key benötigen (für Auto-Fallback)
LOCAL_EMBEDDING_PROVIDERS = {"fastembed", "ollama"}


# ------------------------------------------------------------------ #
# Factory
# ------------------------------------------------------------------ #


def build_embedding(
    provider: str | None = None,
    model: str | None = None,
    **kwargs: Any,
):
    """
    Gibt eine LlamaIndex-kompatible BaseEmbedding-Instanz zurück.

    Auto-Fallback: Wenn provider=None und kein OPENAI_API_KEY gesetzt ist,
    wird automatisch 'fastembed' verwendet.

    Parameter
    ---------
    provider : str | None
        Einer von: openai, fastembed, gemini, mistral, ollama.
        None = Auto-Detect (openai wenn Key vorhanden, sonst fastembed).
    model : str | None
        Modellname. Wenn None, wird der Provider-Default verwendet.
    **kwargs
        Weitere Provider-spezifische Parameter.
    """
    resolved_provider = _resolve_provider(provider)
    defaults = EMBEDDING_DEFAULTS[resolved_provider]
    resolved_model = model or defaults["model"]

    logger.info(
        "Embedding-Factory: provider=%s model=%s", resolved_provider, resolved_model
    )

    if resolved_provider == "openai":
        return _build_openai_embed(resolved_model, **kwargs)
    if resolved_provider == "fastembed":
        return _build_fastembed(resolved_model, **kwargs)
    if resolved_provider == "gemini":
        return _build_gemini_embed(resolved_model, **kwargs)
    if resolved_provider == "mistral":
        return _build_mistral_embed(resolved_model, **kwargs)
    if resolved_provider == "ollama":
        return _build_ollama_embed(resolved_model, **kwargs)


# ------------------------------------------------------------------ #
# Provider-Implementierungen
# ------------------------------------------------------------------ #


def _build_openai_embed(model: str, **kwargs):
    try:
        from llama_index.embeddings.openai import OpenAIEmbedding
    except ImportError:
        raise ImportError(
            "llama-index-embeddings-openai ist nicht installiert. "
            "Installieren: pip install llama-index-embeddings-openai"
        )
    _require_env("OPENAI_API_KEY", "openai (embeddings)")
    return OpenAIEmbedding(model=model, **kwargs)


def _build_fastembed(model: str, **kwargs):
    try:
        from llama_index.embeddings.fastembed import FastEmbedEmbedding
    except ImportError:
        raise ImportError(
            "llama-index-embeddings-fastembed ist nicht installiert. "
            "Installieren: pip install llama-index-embeddings-fastembed"
        )
    return FastEmbedEmbedding(model_name=model, **kwargs)


def _build_gemini_embed(model: str, **kwargs):
    try:
        from llama_index.embeddings.gemini import GeminiEmbedding
    except ImportError:
        raise ImportError(
            "llama-index-embeddings-gemini ist nicht installiert. "
            "Installieren: pip install llama-index-embeddings-gemini"
        )
    _require_env("GOOGLE_API_KEY", "gemini (embeddings)")
    return GeminiEmbedding(model_name=model, **kwargs)


def _build_mistral_embed(model: str, **kwargs):
    try:
        from llama_index.embeddings.mistralai import MistralAIEmbedding
    except ImportError:
        raise ImportError(
            "llama-index-embeddings-mistralai ist nicht installiert. "
            "Installieren: pip install llama-index-embeddings-mistralai"
        )
    _require_env("MISTRAL_API_KEY", "mistral (embeddings)")
    return MistralAIEmbedding(model_name=model, **kwargs)


def _build_ollama_embed(model: str, **kwargs):
    try:
        from llama_index.embeddings.ollama import OllamaEmbedding
    except ImportError:
        raise ImportError(
            "llama-index-embeddings-ollama ist nicht installiert. "
            "Installieren: pip install llama-index-embeddings-ollama"
        )
    base_url = kwargs.pop(
        "base_url", os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    )
    return OllamaEmbedding(model_name=model, base_url=base_url, **kwargs)


# ------------------------------------------------------------------ #
# Hilfsfunktionen
# ------------------------------------------------------------------ #


def _resolve_provider(provider: str | None) -> str:
    """
    Löst den Provider auf. Bei None: Auto-Detect via OPENAI_API_KEY.
    """
    if provider is not None:
        p = provider.lower().strip()
        if p not in VALID_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Unbekannter Embedding-Provider: '{p}'. "
                f"Gültige Provider: {sorted(VALID_EMBEDDING_PROVIDERS)}"
            )
        return p

    # Auto-Detect
    if os.environ.get("OPENAI_API_KEY"):
        logger.info("Embedding-Provider Auto-Detect: openai (OPENAI_API_KEY gesetzt)")
        return "openai"
    logger.info(
        "Embedding-Provider Auto-Detect: fastembed (kein OPENAI_API_KEY – lokales Fallback)"
    )
    return "fastembed"


def _require_env(var: str, provider: str) -> None:
    if not os.environ.get(var):
        raise EnvironmentError(
            f"Umgebungsvariable '{var}' fehlt für Embedding-Provider '{provider}'. "
            f"Bitte in .env oder Shell setzen."
        )


def list_embedding_providers() -> list[str]:
    """Gibt alle unterstützten Embedding-Provider-Namen zurück."""
    return sorted(VALID_EMBEDDING_PROVIDERS)


def default_embedding_model(provider: str) -> str:
    """Gibt das Default-Embedding-Modell für einen Provider zurück."""
    if provider not in EMBEDDING_DEFAULTS:
        raise ValueError(f"Unbekannter Embedding-Provider: '{provider}'")
    return EMBEDDING_DEFAULTS[provider]["model"]


def is_local_provider(provider: str) -> bool:
    """True wenn der Provider keinen API-Key benötigt (lokal)."""
    return provider in LOCAL_EMBEDDING_PROVIDERS
