"""Utilities for Streamlit run-history drift comparison."""

from __future__ import annotations

import logging
from typing import Any

LOGGER = logging.getLogger(__name__)

BEWERTUNG_SEVERITY: dict[str, int] = {
    "konform": 0,
    "teilkonform": 1,
    "nicht_konform": 2,
    "nicht_prüfbar": 3,
    "disputed": 3,
}
UNRESOLVED_STATUSES = {"nicht_prüfbar", "disputed"}


def _to_number(value: Any) -> float:
    return float(value) if isinstance(value, (int, float)) else 0.0


def _severity_for(bewertung: str, prueffeld_id: str, logger: logging.Logger) -> int:
    if bewertung not in BEWERTUNG_SEVERITY:
        logger.warning(
            "Unbekannte Bewertung '%s' für prueffeld '%s'", bewertung, prueffeld_id
        )
    return BEWERTUNG_SEVERITY.get(bewertung, 0)


def build_befund_index(
    report_payload: dict, *, logger: logging.Logger | None = None
) -> dict:
    log = logger or LOGGER
    index = {}
    for sektion in report_payload.get("sektionen", []):
        sid = sektion.get("id", "")
        for befund in sektion.get("befunde", []):
            bid = befund.get("id") or befund.get("prueffeld_id")
            if not bid:
                continue
            if bid in index:
                log.warning(
                    "Doppelte prueffeld_id '%s' in Sektion '%s' (bereits in '%s')",
                    bid,
                    sid,
                    index[bid]["sektion"],
                )
            index[bid] = {
                "sektion": sid,
                "frage": befund.get("frage", ""),
                "bewertung": befund.get("bewertung", ""),
                "confidence": _to_number(befund.get("confidence")),
                "confidence_level": befund.get("confidence_level", ""),
            }
    return index


def build_drift_rows(
    index_a: dict, index_b: dict, *, logger: logging.Logger | None = None
) -> list[dict]:
    log = logger or LOGGER
    rows = []
    keys = sorted(set(index_a.keys()) | set(index_b.keys()))
    for key in keys:
        a = index_a.get(key)
        b = index_b.get(key)
        if a and b:
            bew_a = a.get("bewertung", "")
            bew_b = b.get("bewertung", "")
            if bew_a in UNRESOLVED_STATUSES and bew_b in UNRESOLVED_STATUSES:
                change = "gleich"
            else:
                sev_delta = _severity_for(bew_b, key, log) - _severity_for(
                    bew_a, key, log
                )
                if sev_delta > 0:
                    change = "verschlechtert"
                elif sev_delta < 0:
                    change = "verbessert"
                elif bew_a != bew_b:
                    change = "geändert"
                else:
                    change = "gleich"
            rows.append(
                {
                    "prueffeld_id": key,
                    "sektion": b.get("sektion", a.get("sektion", "")),
                    "frage": b.get("frage", a.get("frage", "")),
                    "bewertung_a": bew_a,
                    "bewertung_b": bew_b,
                    "delta_confidence": round(
                        _to_number(b.get("confidence"))
                        - _to_number(a.get("confidence")),
                        3,
                    ),
                    "status": change,
                }
            )
        elif b and not a:
            rows.append(
                {
                    "prueffeld_id": key,
                    "sektion": b.get("sektion", ""),
                    "frage": b.get("frage", ""),
                    "bewertung_a": "(neu)",
                    "bewertung_b": b.get("bewertung", ""),
                    "delta_confidence": round(_to_number(b.get("confidence")), 3),
                    "status": "neu",
                }
            )
        elif a and not b:
            rows.append(
                {
                    "prueffeld_id": key,
                    "sektion": a.get("sektion", ""),
                    "frage": a.get("frage", ""),
                    "bewertung_a": a.get("bewertung", ""),
                    "bewertung_b": "(entfallen)",
                    "delta_confidence": round(-_to_number(a.get("confidence")), 3),
                    "status": "entfallen",
                }
            )
    return rows
