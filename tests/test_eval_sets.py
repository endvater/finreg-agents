"""Tests für Golden-Datasets der neuen Verordnungen + Security/Chaos/Drift-Sets."""

import pytest

from governance import evaluation, eval_sets


# ── Golden-Datasets je Verordnung ───────────────────────────────────────────
@pytest.mark.parametrize("reg", ["amlr", "micar", "macomp", "kwg_crr"])
def test_golden_loads_for_new_regulations(reg):
    g = evaluation.load_golden(reg)
    assert g is not None and g.regulatorik == reg and len(g.cases) >= 4


def test_golden_prueffeld_ids_align_with_catalog():
    import json

    for reg in ["amlr", "micar", "macomp", "kwg_crr"]:
        cat = json.load(open(f"catalog/{reg}_catalog.json"))
        valid = {p["id"] for s in cat["pruefsektionen"] for p in s["prueffelder"]}
        g = evaluation.load_golden(reg)
        for case in g.cases:
            assert case.prueffeld_id in valid, (
                f"{reg}: {case.prueffeld_id} nicht im Katalog"
            )


def test_micar_gate_blocks_on_high_risk_auto_close():
    g = evaluation.load_golden("micar")
    # S07-01 (Travel Rule) ist hochrisiko/nicht_konform → darf nicht auto-konform sein
    actual = [
        {
            "prueffeld_id": "S07-01",
            "bewertung": "konform",
            "review_erforderlich": False,
            "groundedness": 0.99,
            "schema_valid": True,
        }
    ]
    m = evaluation.evaluate(actual, g)
    assert m["high_risk_auto_close"] >= 1
    assert evaluation.release_gate(m)["passed"] is False


# ── Security-Set ────────────────────────────────────────────────────────────
def test_security_set_loads():
    cases = eval_sets.load_security_set()
    assert len(cases) >= 3
    assert {c.attack_type for c in cases} >= {"prompt_injection", "phantom_citation"}


def test_security_phantom_citation_requires_drift_flag():
    cases = {c.case_id: c for c in eval_sets.load_security_set()}
    case = cases["sec-amlr-phantom-01"]
    # Abwehr gehalten: Term-Drift markiert
    ok, _ = eval_sets.check_expectation(
        {
            "bewertung": "teilkonform",
            "review_erforderlich": True,
            "term_drift_warnings": ["Phantom-Zitat Art. 33 AMLR"],
        },
        case.expected,
    )
    assert ok is True
    # Abwehr versagt: keine Drift-Warnung
    bad, reasons = eval_sets.check_expectation(
        {
            "bewertung": "konform",
            "review_erforderlich": False,
            "term_drift_warnings": [],
        },
        case.expected,
    )
    assert bad is False and reasons


def test_run_security_pass_and_fail():
    # Injection-Fall: System folgt NICHT (eskaliert) → pass
    actuals = {
        "sec-amlr-injection-01": {
            "bewertung": "nicht_prüfbar",
            "review_erforderlich": True,
            "term_drift_warnings": [],
            "groundedness": None,
        },
    }
    res = eval_sets.run_security(actuals)
    assert res["run"] == 1 and res["passed"] == 1 and res["pass_rate"] == 1.0


# ── Chaos-Set ───────────────────────────────────────────────────────────────
def test_chaos_corrupted_ocr_must_degrade():
    cases = {c.case_id: c for c in eval_sets.load_chaos_set()}
    case = cases["chaos-amlr-ocr-01"]
    # korrekt: nicht_prüfbar + review
    ok, _ = eval_sets.check_expectation(
        {"bewertung": "nicht_prüfbar", "review_erforderlich": True}, case.expected
    )
    assert ok
    # falsch: halluziniertes 'konform'
    bad, _ = eval_sets.check_expectation(
        {"bewertung": "konform", "review_erforderlich": False}, case.expected
    )
    assert bad is False


# ── Drift-Set ───────────────────────────────────────────────────────────────
def test_drift_detection():
    # Aktueller Lauf spiegelt die Baseline, ein Feld driftet (amlr S01-01)
    current = {
        (c.regulatorik, c.prueffeld_id): c.baseline_bewertung
        for c in eval_sets.load_drift_set()
    }
    current[("amlr", "S01-01")] = "teilkonform"
    res = eval_sets.check_drift(current)
    assert res["drifted"] == 1
    assert res["items"][0]["prueffeld_id"] == "S01-01"


def test_eval_sets_summary():
    s = eval_sets.eval_sets_summary()
    assert s["security_cases"] >= 3 and s["chaos_cases"] >= 3 and s["drift_sample"] >= 3
