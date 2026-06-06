"""
Block M/O – Minimal-Register (Modell- und Quellenregister).

Leichtgewichtige JSON-gestützte Register als Vorstufe einer Control Plane. Bewusst
schlank gehalten (internes QS-Tool, kein Tier-4-Produktivbetrieb).

  - model_registry.json:  zulässige Modelle/Provider, Freigabestatus, Datenklassen.
  - source_registry.json: ingestierte Quellen mit Owner, Version, Freigabe, Datenklasse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

_REG_DIR = Path(__file__).parent / "registry"


def _load(name: str) -> dict:
    p = _REG_DIR / name
    if not p.exists():
        return {"entries": []}
    return json.loads(p.read_text(encoding="utf-8"))


def _save(name: str, data: dict) -> None:
    _REG_DIR.mkdir(parents=True, exist_ok=True)
    (_REG_DIR / name).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --- Modell-Register --------------------------------------------------------


def list_models(approved_only: bool = False) -> list[dict]:
    entries = _load("model_registry.json").get("entries", [])
    return [e for e in entries if e.get("approved")] if approved_only else entries


def is_model_approved(provider: str, model: Optional[str] = None) -> bool:
    for e in list_models(approved_only=True):
        if e.get("provider") == provider and (
            model is None or e.get("model") in (model, "*")
        ):
            return True
    return False


def provider_concentration() -> dict:
    """Konzentrationsmaß (Anti-Lock-in, Block Q): Anteil je Provider."""
    entries = list_models(approved_only=True)
    counts: dict = {}
    for e in entries:
        counts[e.get("provider")] = counts.get(e.get("provider"), 0) + 1
    total = sum(counts.values()) or 1
    return {p: round(n / total, 4) for p, n in counts.items()}


# --- Quellen-Register -------------------------------------------------------


def register_source(
    *,
    source_id: str,
    owner: str,
    version: str,
    data_class: str = "confidential",
    approved: bool = False,
    sha256: Optional[str] = None,
) -> dict:
    data = _load("source_registry.json")
    entry = {
        "source_id": source_id,
        "owner": owner,
        "version": version,
        "data_class": data_class,
        "approved": approved,
        "sha256": sha256,
    }
    data.setdefault("entries", [])
    data["entries"] = [e for e in data["entries"] if e.get("source_id") != source_id]
    data["entries"].append(entry)
    _save("source_registry.json", data)
    return entry


def list_sources(approved_only: bool = False) -> list[dict]:
    entries = _load("source_registry.json").get("entries", [])
    return [e for e in entries if e.get("approved")] if approved_only else entries


def is_source_approved(source_id: str) -> bool:
    return any(
        e.get("source_id") == source_id and e.get("approved") for e in list_sources()
    )


def registry_summary() -> dict:
    return {
        "models_total": len(list_models()),
        "models_approved": len(list_models(approved_only=True)),
        "provider_concentration": provider_concentration(),
        "sources_total": len(list_sources()),
        "sources_approved": len(list_sources(approved_only=True)),
    }
