"""
Block K – Evaluation / Golden Dataset / Release-Gate.

Größter Hebel für die Vertrauenswürdigkeit der Simulation: Ein versioniertes Golden
Dataset hält fest, welches Prüfer-Urteil je Prüffeld fachlich erwartet wird. Die Eval
vergleicht einen Lauf dagegen und ein Release-Gate blockiert bei Unterschreitung der
Schwellen (Buch Kap. 5).

Golden Datasets liegen in governance/golden/<regulatorik>_*.json. Sie sind absichtlich
als *Seed* angelegt – die Bank füllt sie mit fachlich abgenommenen Fällen
(Standard/Grenz/Hochrisiko), pseudonymisiert.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

_GOLDEN_DIR = Path(__file__).parent / "golden"

# Default-Release-Schwellen (kalibrierbar; Buch nennt Schema≥0.99, Groundedness≥0.97)
DEFAULT_THRESHOLDS = {
    "min_task_success": 0.80,
    "min_schema_compliance": 0.99,
    "min_groundedness": 0.97,
    "min_review_agreement": 0.80,
    "max_high_risk_auto_close": 0.0,  # Hochrisiko nie automatisch 'konform' ohne Review
}


class GoldenCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prueffeld_id: str
    expected_bewertung: str
    expected_review: bool = False
    case_class: str = "standard"  # standard | grenzfall | hochrisiko | fehlerfall
    risk_class: str = "mittel"
    min_groundedness: Optional[float] = None
    notes: str = ""


class GoldenDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    version: str
    regulatorik: str
    origin: str = "seed"  # seed | historisch | synthetisch | incident
    cases: list[GoldenCase] = Field(default_factory=list)


def load_golden(
    regulatorik: str, golden_dir: str | Path | None = None
) -> Optional[GoldenDataset]:
    d = Path(golden_dir) if golden_dir else _GOLDEN_DIR
    matches = sorted(d.glob(f"{regulatorik}_*.json")) if d.exists() else []
    if not matches:
        return None
    raw = json.loads(matches[-1].read_text(encoding="utf-8"))
    # Metafelder (_note o. ä.) ignorieren – Schema ist sonst extra="forbid"
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}
    return GoldenDataset(**raw)


def evaluate(actual_befunde: list[dict], golden: GoldenDataset) -> dict:
    """Vergleicht Lauf-Befunde (dicts) gegen das Golden Dataset.

    actual_befunde: [{prueffeld_id, bewertung, review_erforderlich, confidence,
                      groundedness, schema_valid}, ...]
    """
    by_pf = {b.get("prueffeld_id"): b for b in actual_befunde}
    n = len(golden.cases)
    if n == 0:
        return {"cases": 0, "note": "leeres Golden Dataset (Seed) – keine Aussage"}

    success = agree = grounded_ok = high_risk_violations = matched = 0
    case_results = []
    for case in golden.cases:
        act = by_pf.get(case.prueffeld_id)
        if not act:
            case_results.append(
                {"prueffeld_id": case.prueffeld_id, "status": "missing"}
            )
            continue
        matched += 1
        bew_ok = act.get("bewertung") == case.expected_bewertung
        rev_ok = bool(act.get("review_erforderlich")) == case.expected_review
        gr = act.get("groundedness")
        gr_ok = (case.min_groundedness is None) or (
            gr is not None and gr >= case.min_groundedness
        )
        success += int(bew_ok)
        agree += int(rev_ok)
        grounded_ok += int(gr_ok)
        # Hochrisiko darf nicht ohne Review automatisch 'konform' geschlossen werden
        if (
            case.case_class == "hochrisiko"
            and act.get("bewertung") == "konform"
            and not act.get("review_erforderlich")
        ):
            high_risk_violations += 1
        case_results.append(
            {
                "prueffeld_id": case.prueffeld_id,
                "status": "matched",
                "bewertung_ok": bew_ok,
                "review_ok": rev_ok,
                "groundedness_ok": gr_ok,
            }
        )

    denom = matched or 1
    overall_groundedness = [
        b.get("groundedness")
        for b in actual_befunde
        if b.get("groundedness") is not None
    ]
    schema_vals = [b.get("schema_valid") for b in actual_befunde if "schema_valid" in b]
    return {
        "dataset_id": golden.dataset_id,
        "dataset_version": golden.version,
        "cases": n,
        "matched": matched,
        "missing": n - matched,
        "task_success": round(success / denom, 4),
        "review_agreement": round(agree / denom, 4),
        "groundedness_pass_rate": round(grounded_ok / denom, 4),
        "mean_groundedness": round(
            sum(overall_groundedness) / len(overall_groundedness), 4
        )
        if overall_groundedness
        else None,
        "schema_compliance": round(
            sum(1 for v in schema_vals if v) / len(schema_vals), 4
        )
        if schema_vals
        else None,
        "high_risk_auto_close": high_risk_violations,
        "case_results": case_results,
    }


def release_gate(metrics: dict, thresholds: Optional[dict] = None) -> dict:
    """Entscheidet pass/fail anhand der Schwellen. Liefert Begründung je Kriterium."""
    th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    checks = []

    def check(name, value, op, limit):
        if value is None:
            checks.append(
                {"criterion": name, "value": None, "limit": limit, "passed": None}
            )
            return None
        passed = value >= limit if op == ">=" else value <= limit
        checks.append(
            {
                "criterion": name,
                "value": value,
                "limit": limit,
                "op": op,
                "passed": passed,
            }
        )
        return passed

    check("task_success", metrics.get("task_success"), ">=", th["min_task_success"])
    check(
        "schema_compliance",
        metrics.get("schema_compliance"),
        ">=",
        th["min_schema_compliance"],
    )
    check(
        "groundedness", metrics.get("mean_groundedness"), ">=", th["min_groundedness"]
    )
    check(
        "review_agreement",
        metrics.get("review_agreement"),
        ">=",
        th["min_review_agreement"],
    )
    check(
        "high_risk_auto_close",
        metrics.get("high_risk_auto_close"),
        "<=",
        th["max_high_risk_auto_close"],
    )

    evaluated = [c for c in checks if c["passed"] is not None]
    passed = all(c["passed"] for c in evaluated) if evaluated else False
    return {
        "passed": passed,
        "checks": checks,
        "note": "PASS" if passed else "BLOCKED – Schwelle(n) verletzt oder Eval leer",
    }
