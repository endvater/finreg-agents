"""
Block I – Decision Trace (append-only).

Schreibt je Lauf einen unveränderlichen JSONL-Trace mit Versionen (Modell/Prompt/Katalog),
Provider, Datenklasse, Kosten, Zuständen, Confidence und Review-/Eskalations-Ereignissen.

Bewusst append-only (im Gegensatz zum überschreibenden checkpoint_latest.json) und ohne
Roh-Dokumentinhalte (nur Hashes/IDs) – damit revisionsfähig und datensparsam zugleich.
Liefert die drei Diagnose-Abfragen des Buches: why_flagged / why_not_flagged / what_changed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

TRACE_SCHEMA_VERSION = "1.0"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


class DecisionTrace:
    """Append-only Trace-Writer (eine JSONL-Datei pro Lauf)."""

    def __init__(self, run_id: str, path: str | Path):
        self.run_id = run_id
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False

    def _append(self, event: dict) -> None:
        event = {"ts": _utc(), "run_id": self.run_id, **event}
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")

    # -- Lebenszyklus-Ereignisse --------------------------------------------
    def run_start(
        self,
        *,
        regulatorik: str,
        provider: str,
        model: str,
        catalog_version: str,
        data_class: str = "confidential",
        prompt_version: Optional[str] = None,
        agent_version: Optional[str] = None,
    ) -> None:
        self._append(
            {
                "event": "run_start",
                "schema_version": TRACE_SCHEMA_VERSION,
                "regulatorik": regulatorik,
                "provider": provider,
                "model": model,
                "catalog_version": catalog_version,
                "data_class": data_class,
                "prompt_version": prompt_version,
                "agent_version": agent_version,
            }
        )

    def prueffeld(
        self,
        *,
        prueffeld_id: str,
        sektion_id: str,
        bewertung: str,
        confidence: float,
        review_erforderlich: bool,
        groundedness: Optional[float] = None,
        state: str = "evaluated",
        model: Optional[str] = None,
        routing_reason: Optional[str] = None,
        term_drift_warnings: Optional[list] = None,
        schema_valid: Optional[bool] = None,
    ) -> None:
        self._append(
            {
                "event": "prueffeld",
                "prueffeld_id": prueffeld_id,
                "sektion_id": sektion_id,
                "bewertung": bewertung,
                "confidence": round(float(confidence or 0.0), 4),
                "review_erforderlich": bool(review_erforderlich),
                "groundedness": groundedness,
                "state": state,
                "model": model,
                "routing_reason": routing_reason,
                "term_drift_warnings": term_drift_warnings or [],
                "schema_valid": schema_valid,
            }
        )

    def escalation(
        self, *, sektion_id: str, reason: str, detail: dict | None = None
    ) -> None:
        self._append(
            {
                "event": "escalation",
                "sektion_id": sektion_id,
                "reason": reason,
                "detail": detail or {},
            }
        )

    def run_end(
        self, *, status: str, cost: dict | None = None, metrics: dict | None = None
    ) -> None:
        self._append(
            {
                "event": "run_end",
                "status": status,
                "cost": cost or {},
                "metrics": metrics or {},
            }
        )
        self._closed = True


# ---------------------------------------------------------------------------
# Lese-/Diagnose-Helfer (why_flagged / why_not_flagged / what_changed)
# ---------------------------------------------------------------------------


def read_trace(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    events = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def why_flagged(path: str | Path, prueffeld_id: str) -> Optional[dict]:
    """Warum wurde ein Prüffeld als nicht/teilkonform bzw. review markiert?"""
    for ev in read_trace(path):
        if ev.get("event") == "prueffeld" and ev.get("prueffeld_id") == prueffeld_id:
            return {
                "prueffeld_id": prueffeld_id,
                "bewertung": ev.get("bewertung"),
                "confidence": ev.get("confidence"),
                "review_erforderlich": ev.get("review_erforderlich"),
                "groundedness": ev.get("groundedness"),
                "term_drift_warnings": ev.get("term_drift_warnings"),
                "model": ev.get("model"),
            }
    return None


def why_not_flagged(path: str | Path) -> list[dict]:
    """Welche Prüffelder blieben konform/unmarkiert – mit Confidence/Groundedness?

    Genau die Frage, die in einer Prüfungs-Simulation am ehesten hinterfragt wird:
    'Warum hat die Simulation hier KEINEN Mangel gesehen?'
    """
    out = []
    for ev in read_trace(path):
        if ev.get("event") == "prueffeld" and ev.get("bewertung") == "konform":
            out.append(
                {
                    "prueffeld_id": ev.get("prueffeld_id"),
                    "confidence": ev.get("confidence"),
                    "groundedness": ev.get("groundedness"),
                    "review_erforderlich": ev.get("review_erforderlich"),
                }
            )
    return out


def what_changed(path_a: str | Path, path_b: str | Path) -> list[dict]:
    """Welche Prüffelder unterscheiden sich zwischen zwei Läufen (Drift)?"""

    def index(path):
        idx = {}
        for ev in read_trace(path):
            if ev.get("event") == "prueffeld":
                idx[ev.get("prueffeld_id")] = ev
        return idx

    a, b = index(path_a), index(path_b)
    diffs = []
    for pf in sorted(set(a) | set(b)):
        ea, eb = a.get(pf, {}), b.get(pf, {})
        if ea.get("bewertung") != eb.get("bewertung") or ea.get(
            "review_erforderlich"
        ) != eb.get("review_erforderlich"):
            diffs.append(
                {
                    "prueffeld_id": pf,
                    "von": {
                        "bewertung": ea.get("bewertung"),
                        "confidence": ea.get("confidence"),
                    },
                    "nach": {
                        "bewertung": eb.get("bewertung"),
                        "confidence": eb.get("confidence"),
                    },
                }
            )
    return diffs
