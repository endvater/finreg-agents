"""Tests für das governance/-Paket (QS-/Epistemik-Anforderungen)."""

from types import SimpleNamespace

import pytest

from governance import evidence, schemas, trace, routing, cost, registry, evaluation
from governance.agent_card import load_cards
from governance import monitoring


# ── Block C: Evidence / Provenienz ──────────────────────────────────────────
def test_quote_hash_stable_and_normalized():
    a = evidence.quote_hash("  Der  Kunde   wurde identifiziert. ")
    b = evidence.quote_hash("der kunde wurde identifiziert.")
    assert a == b and a.startswith("sha256:")


def test_evidence_package_groundedness():
    pkg = evidence.EvidencePackage(prueffeld_id="S01-01", claims=[
        evidence.EvidenceClaim("c1", evidence.ClaimStatus.CORROBORATED, quote="x"),
        evidence.EvidenceClaim("c2", evidence.ClaimStatus.UNVERIFIED),
    ])
    assert pkg.groundedness == 0.5
    assert pkg.has_unverified is True
    # quote_hash automatisch gesetzt
    assert pkg.claims[0].quote_hash_value is not None


def test_from_claim_provenance_bridge():
    prov = [SimpleNamespace(claim_text="a", status="corroborated",
                            source_chunk_ids=["n1", "n2"], provenance_id="C001")]
    pkg = evidence.from_claim_provenance("S01-01", prov, catalog_version="2025-02")
    assert pkg.claims[0].status == evidence.ClaimStatus.CORROBORATED
    assert pkg.claims[0].source_version == "2025-02"


# ── Block B: Schema-as-Contract ─────────────────────────────────────────────
def _befund(**kw):
    base = dict(prueffeld_id="S01-01", frage="f?", bewertung="konform",
                begruendung="ok", confidence=0.9, confidence_level="high",
                review_erforderlich=False)
    base.update(kw)
    return SimpleNamespace(**base)


def test_validate_befund_ok():
    assert schemas.validate_befund(_befund()).ok is True


def test_validate_befund_review_rule_enforced():
    # niedrige Confidence ohne review_erforderlich → Vertragsverstoß
    r = schemas.validate_befund(_befund(confidence=0.4, review_erforderlich=False))
    assert r.ok is False and r.errors


def test_validate_befund_bad_bewertung():
    r = schemas.validate_befund(_befund(bewertung="vielleicht"))
    assert r.ok is False


def test_validate_run_aggregates():
    sek = SimpleNamespace(befunde=[_befund(), _befund(bewertung="vielleicht")])
    res = schemas.validate_run([sek])
    assert res["total"] == 2 and res["invalid"] == 1
    assert 0.0 <= res["schema_compliance"] <= 1.0


# ── Block I: Decision Trace ─────────────────────────────────────────────────
def test_trace_append_and_queries(tmp_path):
    p = tmp_path / "decision_trace_run1.jsonl"
    t = trace.DecisionTrace("run1", p)
    t.run_start(regulatorik="gwg", provider="ollama", model="llama3",
                catalog_version="2025-02")
    t.prueffeld(prueffeld_id="S01-01", sektion_id="S01", bewertung="konform",
                confidence=0.9, review_erforderlich=False, groundedness=0.95)
    t.prueffeld(prueffeld_id="S01-02", sektion_id="S01", bewertung="nicht_konform",
                confidence=0.8, review_erforderlich=True)
    t.run_end(status="ok")
    assert len(trace.read_trace(p)) == 4
    assert trace.why_flagged(p, "S01-02")["bewertung"] == "nicht_konform"
    nf = trace.why_not_flagged(p)
    assert len(nf) == 1 and nf[0]["prueffeld_id"] == "S01-01"


def test_trace_what_changed(tmp_path):
    pa, pb = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    ta = trace.DecisionTrace("a", pa)
    ta.prueffeld(prueffeld_id="S01-01", sektion_id="S01", bewertung="konform",
                 confidence=0.9, review_erforderlich=False)
    tb = trace.DecisionTrace("b", pb)
    tb.prueffeld(prueffeld_id="S01-01", sektion_id="S01", bewertung="nicht_konform",
                 confidence=0.7, review_erforderlich=True)
    diffs = trace.what_changed(pa, pb)
    assert diffs and diffs[0]["prueffeld_id"] == "S01-01"


# ── Block J-bis: Routing nach Datenklasse ───────────────────────────────────
def test_routing_confidential_forces_local():
    d = routing.decide_route("confidential", configured_provider="anthropic")
    assert d.requires_local is True
    assert d.provider in routing.LOCAL_PROVIDERS


def test_routing_public_allows_hosted():
    d = routing.decide_route("public", configured_provider="anthropic")
    assert d.requires_local is False and d.provider == "anthropic"


def test_routing_local_provider_kept_for_confidential():
    d = routing.decide_route("internal", configured_provider="ollama")
    assert d.provider == "ollama"


# ── Block J: Kostenmodell ───────────────────────────────────────────────────
def test_hosted_cost_and_cpvct():
    ts = {"nach_agent": {"pruefer": {"input": 10000, "output": 2000}}}
    c = cost.estimate_run_cost(ts, route_is_local=False, valid_tasks=20)
    assert c["mode"] == "hosted" and c["total_cost"] > 0
    assert c["cost_per_valid_task"] == round(c["total_cost"] / 20, 6)


def test_self_hosted_cost_per_run():
    c = cost.estimate_run_cost({"nach_agent": {}}, route_is_local=True, valid_tasks=10)
    assert c["mode"] == "self_hosted" and c["amortized_per_run"] > 0


def test_soft_budget_warn_and_exceed():
    assert cost.soft_budget_check(0.9, budget=1.0)["status"] == "warn"
    assert cost.soft_budget_check(1.2, budget=1.0)["status"] == "exceeded"
    assert cost.soft_budget_check(0.1, budget=1.0)["status"] == "ok"


def test_breakeven_runs():
    assert cost.breakeven_runs(hosted_cost_per_run=10.0) == 150.0


# ── Block M/O: Register ─────────────────────────────────────────────────────
def test_model_registry_approval_and_concentration():
    assert registry.is_model_approved("ollama", "llama3") is True
    assert registry.is_model_approved("gemini") is False  # nicht approved
    conc = registry.provider_concentration()
    assert abs(sum(conc.values()) - 1.0) < 0.01  # Rundung auf 4 Dezimalstellen


# ── Block N: Agent Cards ────────────────────────────────────────────────────
def test_agent_cards_load_and_gattung():
    cards = load_cards()
    names = {c.agent_name: c for c in cards}
    assert "PruefAgent" in names and names["PruefAgent"].gattung == "G2"
    # Konsistenz: G2 ist nicht-entscheidend
    assert names["PruefAgent"].is_decision_making is False
    assert names["PruefAgent"].issues() == []


# ── Block K: Evaluation / Release-Gate ──────────────────────────────────────
def test_load_golden_seed():
    g = evaluation.load_golden("gwg")
    assert g is not None and g.regulatorik == "gwg" and len(g.cases) >= 1


def test_evaluate_and_gate():
    g = evaluation.load_golden("gwg")
    # perfekter Lauf gegen Seed
    actual = [
        {"prueffeld_id": "S01-01", "bewertung": "konform", "review_erforderlich": False,
         "groundedness": 0.95, "schema_valid": True},
        {"prueffeld_id": "S01-02", "bewertung": "nicht_konform", "review_erforderlich": True,
         "groundedness": 0.98, "schema_valid": True},
        {"prueffeld_id": "S01-03", "bewertung": "teilkonform", "review_erforderlich": True,
         "groundedness": 0.98, "schema_valid": True},
    ]
    m = evaluation.evaluate(actual, g)
    assert m["task_success"] == 1.0
    gate = evaluation.release_gate(m)
    assert gate["passed"] is True


def test_gate_blocks_on_high_risk_auto_close():
    g = evaluation.load_golden("gwg")
    actual = [{"prueffeld_id": "S01-02", "bewertung": "konform",
               "review_erforderlich": False, "groundedness": 0.99, "schema_valid": True}]
    m = evaluation.evaluate(actual, g)
    assert m["high_risk_auto_close"] >= 1
    assert evaluation.release_gate(m)["passed"] is False


# ── Monitoring-Aggregation ──────────────────────────────────────────────────
def test_monitoring_summary_and_collect(tmp_path):
    sek = SimpleNamespace(befunde=[
        SimpleNamespace(bewertung=SimpleNamespace(value="konform"), confidence=0.9,
                        review_erforderlich=False, term_drift_warnings=[],
                        prueffeld_id="S01-01", frage="f", begruendung="b",
                        confidence_level="high", belegte_textstellen=[], quellen=[],
                        schweregrad=None, mangel_text=None),
    ])
    summary = monitoring.build_run_summary(
        run_id="r1", regulatorik="gwg", provider="ollama", model="llama3",
        catalog_version="2025-02", sektionsergebnisse=[sek],
        cost={"total_cost": 0.1}, route={"provider": "ollama"})
    monitoring.write_run_summary(summary, tmp_path)
    runs = monitoring.collect_runs(tmp_path)
    assert len(runs) == 1 and runs[0]["run_id"] == "r1"
    snap = monitoring.fleet_snapshot(tmp_path)
    assert snap["runs"] == 1
