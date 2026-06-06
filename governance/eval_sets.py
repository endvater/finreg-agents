"""
Eval-Splits jenseits des Golden Sets: Security, Chaos, Drift (Playbook Kap. 5/10).

Diese Sets prüfen nicht die fachliche Trefferquote (das macht das Golden Set), sondern
das *Verhalten unter Angriff/Störung*:

  - Security-Set: eingebettete Prompt-Injection, RAG-Poisoning, Phantom-Zitate,
    widersprüchliche Quellen → das System muss markieren/eskalieren, nicht still folgen.
  - Chaos-Set: beschädigtes OCR, fehlende Dokumente, widersprüchliche Zahlen, Überlänge
    → das System muss kontrolliert auf nicht_prüfbar/Review gehen, nicht halluzinieren.
  - Drift-Set: wiederkehrende Stichprobe mit Baseline-Urteil → Abweichung = Drift-Signal.

Die Fixtures (governance/eval_sets/*.json) sind *Spezifikationen* mit erwartetem
Verhalten. Die Prüf-Funktionen bewerten einen tatsächlichen Lauf (Befund-Dicts) dagegen.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict

_EVAL_DIR = Path(__file__).parent / "eval_sets"


class EvalExpectation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bewertung_in: Optional[list[str]] = None
    must_review: Optional[bool] = None
    must_flag_term_drift: Optional[bool] = None
    max_groundedness: Optional[float] = None
    min_groundedness: Optional[float] = None


class SecurityCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    attack_type: str  # prompt_injection | rag_poisoning | phantom_citation | contradictory_sources
    regulatorik: str
    prueffeld_id: str
    injected_text: str = ""
    expected: EvalExpectation
    notes: str = ""


class ChaosCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str
    chaos_type: str  # corrupted_ocr | missing_document | contradictory_numbers | oversized_input
    regulatorik: str
    prueffeld_id: str
    expected: EvalExpectation
    notes: str = ""


class DriftCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    regulatorik: str
    prueffeld_id: str
    baseline_bewertung: str
    notes: str = ""


def _load(name: str) -> dict:
    p = _EVAL_DIR / name
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {k: v for k, v in raw.items() if not k.startswith("_")}


def load_security_set() -> list[SecurityCase]:
    return [SecurityCase(**c) for c in _load("security_seed.json").get("cases", [])]


def load_chaos_set() -> list[ChaosCase]:
    return [ChaosCase(**c) for c in _load("chaos_seed.json").get("cases", [])]


def load_drift_set() -> list[DriftCase]:
    return [DriftCase(**c) for c in _load("drift_seed.json").get("cases", [])]


# ---------------------------------------------------------------------------
# Prüf-Funktionen (bewerten einen tatsächlichen Lauf gegen die Erwartung)
# ---------------------------------------------------------------------------


def check_expectation(actual: dict, exp: EvalExpectation) -> tuple[bool, list[str]]:
    """actual: {bewertung, review_erforderlich, term_drift_warnings, groundedness}."""
    reasons: list[str] = []
    if exp.bewertung_in is not None and actual.get("bewertung") not in exp.bewertung_in:
        reasons.append(
            f"bewertung {actual.get('bewertung')!r} nicht in {exp.bewertung_in}"
        )
    if (
        exp.must_review is not None
        and bool(actual.get("review_erforderlich")) != exp.must_review
    ):
        reasons.append(f"review_erforderlich != {exp.must_review}")
    if exp.must_flag_term_drift:
        if not (actual.get("term_drift_warnings") or []):
            reasons.append("term_drift_warning erwartet, aber keine vorhanden")
    gnd = actual.get("groundedness")
    if (
        exp.max_groundedness is not None
        and gnd is not None
        and gnd > exp.max_groundedness
    ):
        reasons.append(f"groundedness {gnd} > erlaubt {exp.max_groundedness}")
    if (
        exp.min_groundedness is not None
        and gnd is not None
        and gnd < exp.min_groundedness
    ):
        reasons.append(f"groundedness {gnd} < erforderlich {exp.min_groundedness}")
    return (len(reasons) == 0, reasons)


def run_security(actuals_by_case: dict) -> dict:
    """actuals_by_case: {case_id: actual_befund_dict}. Fehlende Fälle = nicht ausgeführt."""
    cases = load_security_set()
    results, passed = [], 0
    for c in cases:
        act = actuals_by_case.get(c.case_id)
        if act is None:
            results.append({"case_id": c.case_id, "status": "not_run"})
            continue
        ok, reasons = check_expectation(act, c.expected)
        passed += int(ok)
        results.append(
            {
                "case_id": c.case_id,
                "attack_type": c.attack_type,
                "passed": ok,
                "reasons": reasons,
            }
        )
    run = [r for r in results if r.get("status") != "not_run"]
    return {
        "total": len(cases),
        "run": len(run),
        "passed": passed,
        "pass_rate": round(passed / len(run), 4) if run else None,
        "results": results,
    }


def run_chaos(actuals_by_case: dict) -> dict:
    cases = load_chaos_set()
    results, passed = [], 0
    for c in cases:
        act = actuals_by_case.get(c.case_id)
        if act is None:
            results.append({"case_id": c.case_id, "status": "not_run"})
            continue
        ok, reasons = check_expectation(act, c.expected)
        passed += int(ok)
        results.append(
            {
                "case_id": c.case_id,
                "chaos_type": c.chaos_type,
                "passed": ok,
                "reasons": reasons,
            }
        )
    run = [r for r in results if r.get("status") != "not_run"]
    return {
        "total": len(cases),
        "run": len(run),
        "passed": passed,
        "pass_rate": round(passed / len(run), 4) if run else None,
        "results": results,
    }


def check_drift(current_by_pf: dict) -> dict:
    """current_by_pf: {(regulatorik, prueffeld_id) | prueffeld_id: bewertung}.

    Vergleicht die wiederkehrende Stichprobe gegen ihr Baseline-Urteil.
    """
    cases = load_drift_set()
    drifted = []
    for c in cases:
        key = (c.regulatorik, c.prueffeld_id)
        cur = current_by_pf.get(key, current_by_pf.get(c.prueffeld_id))
        if cur is not None and cur != c.baseline_bewertung:
            drifted.append(
                {
                    "regulatorik": c.regulatorik,
                    "prueffeld_id": c.prueffeld_id,
                    "baseline": c.baseline_bewertung,
                    "current": cur,
                }
            )
    return {"sample_size": len(cases), "drifted": len(drifted), "items": drifted}


def eval_sets_summary() -> dict:
    return {
        "security_cases": len(load_security_set()),
        "chaos_cases": len(load_chaos_set()),
        "drift_sample": len(load_drift_set()),
    }
