"""
FinRegAgents – Governance-Paket

Setzt die für FinRegAgents *proportional sinnvollen* Anforderungen aus
„Thinking Agentic – das Playbook" um (siehe docs/anforderungen-thinking-agentic.md).

FinRegAgents ist ein internes Simulations-/QS-Werkzeug (keine bindenden Entscheidungen,
keine Tool-Ausführung, human-reviewed). Umgesetzt wird daher der QS-/Epistemik-Kern:

  Block B  – Schema-as-Contract            → schemas.py
  Block C  – EvidencePackage/Provenienz    → evidence.py
  Block I  – Decision Trace (append-only)  → trace.py
  Block J  – Routing nach Datenklasse      → routing.py
  Block J  – Kostenmodell inkl. Self-Host  → cost.py
  Block K  – Eval / Golden Dataset / Gate  → evaluation.py
  Block M/O– Quellen-/Modell-Register      → registry.py
  Block N  – Agent Card / Versionierung    → agent_card.py
  Monitoring-Aggregation für das Dashboard → monitoring.py

Bewusst NICHT umgesetzt (Über-Engineering für ein internes QS-Tool, Gattung G1):
  Block F  – Tool-Governance/MCP (keine Tools)
  Block J  – Budget-Circuit-Breaker als Safety-Hard-Stop
  Kill-Switch-Stufen, Three-Lines-Control-Plane in Vollausbau.
"""

GOVERNANCE_VERSION = "0.1.0"

__all__ = ["GOVERNANCE_VERSION"]
