"""
Block J(-bis) – Routing nach Datenklasse (Kosten + Datenhoheit in einem).

Buch-Regel: „Zulässigkeit vor Kostenoptimierung." Vertrauliche Bankdokumente dürfen ein
fremdgehostetes LLM nicht erreichen (Auslagerung/DSGVO) → lokales Modell (Ollama/fastembed).
Öffentlicher/Kataloginhalt darf fremdgehostet verarbeitet werden. Erst innerhalb der
zulässigen Kandidaten wird nach Kosten optimiert.

Die Policy ist versioniert in governance/policies/routing_policy.json hinterlegt.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_POLICY_PATH = Path(__file__).parent / "policies" / "routing_policy.json"

# Provider, die lokal/selbstgehostet laufen (Daten verlassen das Haus nicht)
LOCAL_PROVIDERS = {"ollama", "fastembed"}


@dataclass
class RoutingDecision:
    data_class: str
    risk_class: str
    requires_local: bool
    provider: str
    model: Optional[str]
    reason: str
    policy_version: str


def load_policy(path: str | Path | None = None) -> dict:
    p = Path(path) if path else _POLICY_PATH
    if not p.exists():
        return _DEFAULT_POLICY
    return json.loads(p.read_text(encoding="utf-8"))


_DEFAULT_POLICY = {
    "policy_version": "2026-06",
    "data_classes": {
        "customer": {"requires_local": True},
        "confidential": {"requires_local": True},
        "internal": {"requires_local": True},
        "public": {"requires_local": False},
        "catalog": {"requires_local": False},
    },
    "local": {"provider": "ollama", "model": None},
    "hosted": {"provider": "anthropic", "model": None},
    # Risikoklassen, die trotz Self-Hosting ein stärkeres (ggf. hosted) Modell rechtfertigen
    "escalate_risk_to_hosted": [],
}


def decide_route(
    data_class: str,
    *,
    risk_class: str = "mittel",
    configured_provider: Optional[str] = None,
    policy: Optional[dict] = None,
) -> RoutingDecision:
    """Entscheidet provider/model für eine Datenklasse.

    configured_provider: der vom Nutzer gewählte Provider (z. B. CLI --provider).
    Bei vertraulichen Daten wird ein fremdgehosteter Provider auf den lokalen Pfad
    *gezwungen* und der Grund dokumentiert (Routing-Ereignis für den Trace).
    """
    policy = policy or load_policy()
    dc = (data_class or "confidential").lower()
    dc_cfg = policy.get("data_classes", {}).get(dc, {"requires_local": True})
    requires_local = bool(dc_cfg.get("requires_local", True))

    local = policy.get("local", _DEFAULT_POLICY["local"])
    hosted = policy.get("hosted", _DEFAULT_POLICY["hosted"])
    pv = policy.get("policy_version", "unknown")

    if requires_local:
        configured_is_local = (configured_provider or "") in LOCAL_PROVIDERS
        if configured_is_local:
            return RoutingDecision(
                dc,
                risk_class,
                True,
                configured_provider,
                None,
                f"Datenklasse '{dc}' verlangt lokale Verarbeitung; konfigurierter "
                f"Provider '{configured_provider}' ist lokal – zulässig.",
                pv,
            )
        return RoutingDecision(
            dc,
            risk_class,
            True,
            local["provider"],
            local.get("model"),
            f"Datenklasse '{dc}' verlangt lokale Verarbeitung (Datenhoheit/DSGVO); "
            f"erzwinge lokalen Provider '{local['provider']}' statt "
            f"fremdgehostet '{configured_provider}'.",
            pv,
        )

    # Öffentlich/Katalog: fremdgehostet zulässig, Kostenoptimierung erlaubt
    provider = configured_provider or hosted["provider"]
    return RoutingDecision(
        dc,
        risk_class,
        False,
        provider,
        hosted.get("model"),
        f"Datenklasse '{dc}' erlaubt fremdgehostete Verarbeitung; nutze '{provider}'.",
        pv,
    )
