"""
Gemeinsame Test-Fixtures für FinRegAgents.

ANTHROPIC_API_KEY wird automatisch als Dummy gesetzt, damit Tests,
die SkeptikerAgent / PrueferAgent instanziieren, keinen echten Key
benötigen. Die LLM-Methoden werden in den jeweiligen Tests separat
gemockt, wenn LLM-Aufrufe relevant sind.
"""

import pytest


@pytest.fixture(autouse=True)
def mock_anthropic_key(monkeypatch):
    """Setzt ANTHROPIC_API_KEY als Dummy, falls nicht gesetzt."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key-for-testing")
