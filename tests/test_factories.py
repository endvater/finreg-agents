"""
Tests für LLM-Factory und Embedding-Factory (agents/llm_factory.py + agents/embedding_factory.py)
Ausführen: pytest tests/test_factories.py -v
"""

import os
import pytest
from unittest.mock import MagicMock, patch


# ------------------------------------------------------------------ #
# LLM Factory
# ------------------------------------------------------------------ #

from agents.llm_factory import (
    build_llm,
    list_providers,
    default_model,
    PROVIDER_DEFAULTS,
    VALID_PROVIDERS,
)


class TestLLMFactory:

    def test_list_providers_returns_all_seven(self):
        providers = list_providers()
        assert set(providers) == {
            "anthropic", "openai", "gemini", "mistral", "cohere", "grok", "ollama"
        }

    def test_default_model_known_provider(self):
        assert default_model("anthropic") == PROVIDER_DEFAULTS["anthropic"]["model"]
        assert default_model("openai") == PROVIDER_DEFAULTS["openai"]["model"]
        assert default_model("gemini") == PROVIDER_DEFAULTS["gemini"]["model"]
        assert default_model("ollama") == PROVIDER_DEFAULTS["ollama"]["model"]

    def test_default_model_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unbekannter Provider"):
            default_model("nonexistent")

    def test_build_llm_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unbekannter Provider"):
            build_llm("gpt99")

    def test_build_llm_missing_env_raises(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="ANTHROPIC_API_KEY"):
            build_llm("anthropic")

    def test_build_llm_anthropic_uses_default_model(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
        mock_cls = MagicMock()
        with patch("agents.llm_factory._build_anthropic", return_value=mock_cls) as m:
            build_llm("anthropic")
            m.assert_called_once()
            # Kein explizites Modell → Default aus PROVIDER_DEFAULTS
            args, kwargs = m.call_args
            assert args[0] == PROVIDER_DEFAULTS["anthropic"]["model"]

    def test_build_llm_openai_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        # Paket-Import mocken damit nur der Key-Check getestet wird
        with patch.dict("sys.modules", {"langchain_openai": MagicMock()}):
            with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
                build_llm("openai")

    def test_build_llm_gemini_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        with patch.dict("sys.modules", {"langchain_google_genai": MagicMock()}):
            with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
                build_llm("gemini")

    def test_build_llm_mistral_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        with patch.dict("sys.modules", {"langchain_mistralai": MagicMock()}):
            with pytest.raises(EnvironmentError, match="MISTRAL_API_KEY"):
                build_llm("mistral")

    def test_build_llm_cohere_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("COHERE_API_KEY", raising=False)
        with patch.dict("sys.modules", {"langchain_cohere": MagicMock()}):
            with pytest.raises(EnvironmentError, match="COHERE_API_KEY"):
                build_llm("cohere")

    def test_build_llm_grok_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("XAI_API_KEY", raising=False)
        with patch.dict("sys.modules", {"langchain_openai": MagicMock()}):
            with pytest.raises(EnvironmentError, match="XAI_API_KEY"):
                build_llm("grok")

    def test_build_llm_ollama_no_key_required(self, monkeypatch):
        """Ollama benötigt keinen API-Key – nur das Paket muss installiert sein."""
        mock_ollama = MagicMock()
        with patch("agents.llm_factory._build_ollama", return_value=mock_ollama):
            result = build_llm("ollama", model="llama3.3")
            assert result is mock_ollama

    def test_build_llm_ollama_warns_on_small_model(self, monkeypatch, caplog):
        """Warnung bei Modellen unter der Mindestempfehlung."""
        import logging
        mock_ollama_module = MagicMock()
        mock_ollama_module.ChatOllama = MagicMock(return_value=MagicMock())
        with patch.dict("sys.modules", {"langchain_ollama": mock_ollama_module}):
            with caplog.at_level(logging.WARNING, logger="agents.llm_factory"):
                build_llm("ollama", model="tinyllama")
        assert any("tinyllama" in r.message for r in caplog.records)

    def test_build_llm_grok_uses_xai_base_url(self, monkeypatch):
        monkeypatch.setenv("XAI_API_KEY", "xai-test")
        mock_openai_module = MagicMock()
        captured_kwargs = {}

        def capture(**kwargs):
            captured_kwargs.update(kwargs)
            return MagicMock()

        mock_openai_module.ChatOpenAI = capture
        with patch.dict("sys.modules", {"langchain_openai": mock_openai_module}):
            build_llm("grok")
        assert "x.ai" in captured_kwargs.get("base_url", "")

    def test_provider_defaults_have_required_keys(self):
        for provider, defaults in PROVIDER_DEFAULTS.items():
            assert "model" in defaults, f"{provider}: 'model' fehlt in PROVIDER_DEFAULTS"


# ------------------------------------------------------------------ #
# Embedding Factory
# ------------------------------------------------------------------ #

from agents.embedding_factory import (
    build_embedding,
    list_embedding_providers,
    default_embedding_model,
    is_local_provider,
    EMBEDDING_DEFAULTS,
    VALID_EMBEDDING_PROVIDERS,
)


class TestEmbeddingFactory:

    def test_list_embedding_providers_returns_all_five(self):
        providers = list_embedding_providers()
        assert set(providers) == {"openai", "fastembed", "gemini", "mistral", "ollama"}

    def test_default_embedding_model_known_providers(self):
        assert default_embedding_model("openai") == "text-embedding-3-small"
        assert default_embedding_model("fastembed") == "BAAI/bge-small-en-v1.5"
        assert default_embedding_model("gemini") == "text-embedding-004"
        assert default_embedding_model("mistral") == "mistral-embed"
        assert default_embedding_model("ollama") == "nomic-embed-text"

    def test_default_embedding_model_unknown_raises(self):
        with pytest.raises(ValueError, match="Unbekannter Embedding-Provider"):
            default_embedding_model("nonexistent")

    def test_is_local_provider(self):
        assert is_local_provider("fastembed") is True
        assert is_local_provider("ollama") is True
        assert is_local_provider("openai") is False
        assert is_local_provider("gemini") is False
        assert is_local_provider("mistral") is False

    def test_build_embedding_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unbekannter Embedding-Provider"):
            build_embedding(provider="xyz")

    def test_build_embedding_auto_detect_uses_fastembed_without_openai_key(
        self, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        mock_embed = MagicMock()
        with patch("agents.embedding_factory._build_fastembed", return_value=mock_embed) as m:
            result = build_embedding(provider=None)
            m.assert_called_once()
            assert result is mock_embed

    def test_build_embedding_auto_detect_uses_openai_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_embed = MagicMock()
        with patch("agents.embedding_factory._build_openai_embed", return_value=mock_embed) as m:
            result = build_embedding(provider=None)
            m.assert_called_once()
            assert result is mock_embed

    def test_build_embedding_openai_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            build_embedding(provider="openai")

    def test_build_embedding_gemini_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        mock_gemini_module = MagicMock()
        mock_gemini_module.GeminiEmbedding = MagicMock()
        with patch.dict("sys.modules", {"llama_index.embeddings.gemini": mock_gemini_module}):
            with pytest.raises(EnvironmentError, match="GOOGLE_API_KEY"):
                build_embedding(provider="gemini")

    def test_build_embedding_mistral_missing_key_raises(self, monkeypatch):
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        mock_mistral_module = MagicMock()
        mock_mistral_module.MistralAIEmbedding = MagicMock()
        with patch.dict("sys.modules", {"llama_index.embeddings.mistralai": mock_mistral_module}):
            with pytest.raises(EnvironmentError, match="MISTRAL_API_KEY"):
                build_embedding(provider="mistral")

    def test_build_embedding_fastembed_no_key_required(self):
        mock_embed = MagicMock()
        with patch("agents.embedding_factory._build_fastembed", return_value=mock_embed):
            result = build_embedding(provider="fastembed")
            assert result is mock_embed

    def test_build_embedding_ollama_no_key_required(self):
        mock_embed = MagicMock()
        with patch("agents.embedding_factory._build_ollama_embed", return_value=mock_embed):
            result = build_embedding(provider="ollama")
            assert result is mock_embed

    def test_build_embedding_custom_model_passed_through(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        mock_embed = MagicMock()
        with patch(
            "agents.embedding_factory._build_openai_embed", return_value=mock_embed
        ) as m:
            build_embedding(provider="openai", model="text-embedding-3-large")
            args, _ = m.call_args
            assert args[0] == "text-embedding-3-large"

    def test_embedding_defaults_have_model_key(self):
        for provider, defaults in EMBEDDING_DEFAULTS.items():
            assert "model" in defaults, f"{provider}: 'model' fehlt in EMBEDDING_DEFAULTS"
