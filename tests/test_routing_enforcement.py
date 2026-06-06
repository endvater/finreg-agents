"""Tests für die Routing-Durchsetzung in der Pipeline (Block J-bis, Enforcement)."""

from pipeline import AuditPipeline


def _pipe(**kw):
    base = dict(input_dir="./demo", regulatorik="gwg", provider="anthropic")
    base.update(kw)
    return AuditPipeline(**base)


def test_enforce_confidential_forces_local_llm_and_embeddings():
    p = _pipe(data_class="confidential", enforce_routing=True,
              embedding_provider="openai")
    p._resolve_routing()
    assert p.provider == "ollama"          # LLM lokal erzwungen
    assert p.embedding_provider == "fastembed"  # Embeddings lokal erzwungen
    assert p._route_enforced is True


def test_no_enforce_keeps_configured_provider():
    p = _pipe(data_class="confidential", enforce_routing=False,
              embedding_provider="openai")
    p._resolve_routing()
    assert p.provider == "anthropic"       # nur Monitoring, kein Override
    assert p.embedding_provider == "openai"
    assert p._route_enforced is False
    # Routing-Entscheidung trotzdem erfasst (advisory)
    assert p._route is not None and p._route.requires_local is True


def test_public_data_class_allows_hosted_even_with_enforce():
    p = _pipe(data_class="public", enforce_routing=True, embedding_provider="openai")
    p._resolve_routing()
    assert p.provider == "anthropic"
    assert p.embedding_provider == "openai"
    assert p._route_enforced is False


def test_enforce_keeps_already_local_provider():
    p = _pipe(provider="ollama", data_class="confidential", enforce_routing=True)
    p._resolve_routing()
    assert p.provider == "ollama"
    # bereits lokal → kein erzwungener Wechsel nötig
    assert p._route_enforced is False
