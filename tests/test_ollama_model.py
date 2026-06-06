"""Tests für die OLLAMA_MODEL-Auflösung (lokales Modell global wählbar, z. B. gemma)."""

from agents.llm_factory import default_model


def test_default_ollama_without_env(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL", raising=False)
    assert default_model("ollama") == "llama3.3"


def test_ollama_model_env_overrides_default(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:1b")
    assert default_model("ollama") == "gemma3:1b"


def test_ollama_model_env_does_not_affect_other_providers(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:1b")
    # Andere Provider bleiben unberührt
    assert default_model("anthropic") != "gemma3:1b"
