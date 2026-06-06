"""
Block N – Agent Card / Versionierung.

Lädt und validiert Agent Cards (Steckbriefe) aus governance/agent_cards/*.json.
Eine Agent Card dokumentiert Zweck, Nicht-Zweck, Gattung, Risikoklasse, Owner, erlaubte
Datenklassen, Modelle, Human-Gates und Versionen – das Pflicht-Artefakt aus Kap. 1/11.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_CARD_DIR = Path(__file__).parent / "agent_cards"

VALID_GENERA = {"G1", "G2", "G3", "G4", "G5"}
VALID_RISK = {"niedrig", "mittel", "hoch", "kritisch"}


class AgentCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent_name: str
    version: str
    gattung: str = Field(description="G1–G5 gemäß Gattungs-Matrix")
    purpose: str
    non_purpose: list[str] = Field(default_factory=list)
    risk_class: str
    business_owner: str = "unbenannt"
    technical_owner: str = "unbenannt"
    allowed_data_classes: list[str] = Field(default_factory=list)
    approved_models: list[str] = Field(default_factory=list)
    human_gate_states: list[str] = Field(default_factory=list)
    prompt_version: str = "n/a"
    is_decision_making: bool = False
    calls_tools: bool = False

    def issues(self) -> list[str]:
        """Konsistenz-Checks (liefert Warnungen, wirft nicht)."""
        out = []
        if self.gattung not in VALID_GENERA:
            out.append(f"unbekannte Gattung: {self.gattung}")
        if self.risk_class not in VALID_RISK:
            out.append(f"unbekannte Risikoklasse: {self.risk_class}")
        if not self.non_purpose:
            out.append("non_purpose leer – Nicht-Zweck sollte explizit sein")
        # Augenmaß-Regel: ein internes Simulations-Tool sollte nicht 'bindend entscheiden'
        if self.is_decision_making and self.gattung != "G1":
            out.append("is_decision_making=True, aber Gattung != G1 – widersprüchlich")
        return out


def load_cards(card_dir: str | Path | None = None) -> list[AgentCard]:
    d = Path(card_dir) if card_dir else _CARD_DIR
    if not d.exists():
        return []
    cards = []
    for f in sorted(d.glob("*.json")):
        cards.append(AgentCard(**json.loads(f.read_text(encoding="utf-8"))))
    return cards


def cards_summary(card_dir: str | Path | None = None) -> dict:
    cards = load_cards(card_dir)
    return {
        "total": len(cards),
        "by_gattung": {
            g: sum(1 for c in cards if c.gattung == g)
            for g in sorted(VALID_GENERA)
            if any(c.gattung == g for c in cards)
        },
        "with_issues": {c.agent_name: c.issues() for c in cards if c.issues()},
    }
