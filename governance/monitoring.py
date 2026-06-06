"""
Monitoring-Aggregation für das Überwachungs-Dashboard.

Baut je Lauf eine kompakte `governance_summary` und sammelt diese über mehrere Läufe
hinweg ein (für das Streamlit-Dashboard und ein CLI-Snapshot). Liest ausschließlich
abgelegte Artefakte – keine Abhängigkeit zum Modell-Stack.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from governance.schemas import validate_run


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_summary(
    *,
    run_id: str,
    regulatorik: str,
    provider: str,
    model: str,
    catalog_version: str,
    sektionsergebnisse,
    cost: dict,
    route: Optional[dict] = None,
    eval_result: Optional[dict] = None,
    gate_result: Optional[dict] = None,
) -> dict:
    """Verdichtet einen Lauf zu einer Monitoring-Summary (JSON-serialisierbar)."""
    befunde = [b for s in sektionsergebnisse for b in getattr(s, "befunde", [])]
    n = len(befunde) or 1
    conf = [float(getattr(b, "confidence", 0.0) or 0.0) for b in befunde]
    bew_counts: dict = {}
    for b in befunde:
        key = getattr(
            getattr(b, "bewertung", None), "value", getattr(b, "bewertung", "?")
        )
        bew_counts[key] = bew_counts.get(key, 0) + 1
    review_n = sum(1 for b in befunde if getattr(b, "review_erforderlich", False))
    disputed_n = bew_counts.get("disputed", 0)
    drift_n = sum(1 for b in befunde if getattr(b, "term_drift_warnings", []))
    schema = validate_run(sektionsergebnisse)
    valid_tasks = sum(
        1
        for b in befunde
        if getattr(getattr(b, "bewertung", None), "value", getattr(b, "bewertung", ""))
        != "nicht_prüfbar"
    )

    return {
        "run_id": run_id,
        "timestamp": _utc(),
        "regulatorik": regulatorik,
        "provider": provider,
        "model": model,
        "catalog_version": catalog_version,
        "befunde_total": len(befunde),
        "bewertung_counts": bew_counts,
        "review_rate": round(review_n / n, 4),
        "disputed_count": disputed_n,
        "term_drift_count": drift_n,
        "confidence_mean": round(sum(conf) / len(conf), 4) if conf else 0.0,
        "confidence_min": round(min(conf), 4) if conf else 0.0,
        "schema_compliance": schema["schema_compliance"],
        "schema_violations": len(schema["violations"]),
        "valid_tasks": valid_tasks,
        "cost": cost,
        "route": route or {},
        "eval": eval_result or {},
        "gate": gate_result or {},
    }


def write_run_summary(summary: dict, output_dir: str | Path) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"governance_summary_{summary['run_id']}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    # Latest-Pointer für einfache Anzeige
    (out / "governance_summary_latest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return str(path)


def collect_runs(output_dir: str | Path) -> list[dict]:
    """Liest alle governance_summary_*.json eines Verzeichnisses (neueste zuerst)."""
    out = Path(output_dir)
    if not out.exists():
        return []
    runs = []
    for f in out.glob("governance_summary_*.json"):
        if f.name == "governance_summary_latest.json":
            continue
        try:
            runs.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return sorted(runs, key=lambda r: r.get("timestamp", ""), reverse=True)


def fleet_snapshot(output_dir: str | Path) -> dict:
    """Aggregat über alle Läufe – die KPI-Kacheln des Dashboards."""
    runs = collect_runs(output_dir)
    if not runs:
        return {"runs": 0}
    n = len(runs)
    gates = [r.get("gate", {}).get("passed") for r in runs if r.get("gate")]
    return {
        "runs": n,
        "regulatoriken": sorted({r.get("regulatorik") for r in runs}),
        "mean_review_rate": round(sum(r.get("review_rate", 0) for r in runs) / n, 4),
        "mean_schema_compliance": round(
            sum(r.get("schema_compliance", 0) for r in runs) / n, 4
        ),
        "mean_confidence": round(sum(r.get("confidence_mean", 0) for r in runs) / n, 4),
        "total_disputed": sum(r.get("disputed_count", 0) for r in runs),
        "total_term_drift": sum(r.get("term_drift_count", 0) for r in runs),
        "gate_pass_rate": round(sum(1 for g in gates if g) / len(gates), 4)
        if gates
        else None,
        "latest": runs[0],
    }


def cli_snapshot(output_dir: str | Path = "./reports/output") -> str:
    """Text-Snapshot für die Konsole (ohne Streamlit)."""
    snap = fleet_snapshot(output_dir)
    if not snap.get("runs"):
        return "Keine Läufe gefunden in " + str(output_dir)
    lines = [
        f"Läufe:               {snap['runs']}",
        f"Regulatoriken:       {', '.join(snap['regulatoriken'])}",
        f"Ø Review-Quote:      {snap['mean_review_rate']:.0%}",
        f"Ø Schema-Compliance: {snap['mean_schema_compliance']:.0%}",
        f"Ø Confidence:        {snap['mean_confidence']:.2f}",
        f"Disputed gesamt:     {snap['total_disputed']}",
        f"Term-Drift gesamt:   {snap['total_term_drift']}",
        f"Release-Gate Pass:   {snap['gate_pass_rate']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    print(cli_snapshot(sys.argv[1] if len(sys.argv) > 1 else "./reports/output"))
