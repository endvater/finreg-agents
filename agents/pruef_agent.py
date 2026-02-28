"""
GwG Audit Pipeline – Prüfer-Agent
Ein Agent pro Prüffeld: RAG-Suche → Bewertung → Befund
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json

from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.retrievers import VectorIndexRetriever
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


# ------------------------------------------------------------------ #
# Datenmodelle
# ------------------------------------------------------------------ #

class Bewertung(str, Enum):
    KONFORM = "konform"
    TEILKONFORM = "teilkonform"
    NICHT_KONFORM = "nicht_konform"
    NICHT_PRUEFBAR = "nicht_prüfbar"  # Keine ausreichende Evidenz im Index


class Schweregrad(str, Enum):
    WESENTLICH = "wesentlich"
    BEDEUTSAM = "bedeutsam"
    GERING = "gering"


@dataclass
class Befund:
    prueffeld_id: str
    frage: str
    bewertung: Bewertung
    begruendung: str
    belegte_textstellen: list[str] = field(default_factory=list)
    empfehlungen: list[str] = field(default_factory=list)
    mangel_text: Optional[str] = None
    schweregrad: Optional[str] = None
    quellen: list[str] = field(default_factory=list)


@dataclass
class SektionsergebniS:
    sektion_id: str
    titel: str
    befunde: list[Befund] = field(default_factory=list)

    @property
    def kritische_befunde(self) -> list[Befund]:
        return [b for b in self.befunde
                if b.bewertung in (Bewertung.NICHT_KONFORM, Bewertung.TEILKONFORM)]


# ------------------------------------------------------------------ #
# Prüfer-Agent
# ------------------------------------------------------------------ #

SYSTEM_PROMPT_PRUEFER = """
Du bist ein erfahrener Sonderprüfer der BaFin mit Spezialisierung auf Geldwäscheprävention (GwG).
Du führst eine Sonderprüfung gemäß §25h KWG und GwG durch.

Deine Aufgabe:
1. Analysiere die bereitgestellten Dokumentenausschnitte (Evidenz)
2. Beantworte die Prüffrage präzise und mit direktem Bezug auf die Evidenz
3. Bewerte gemäß: konform | teilkonform | nicht_konform | nicht_prüfbar
4. Belege deine Bewertung mit konkreten Textstellen aus den Dokumenten
5. Formuliere ggf. einen Mangel im BaFin-Stil und konkrete Empfehlungen

Wichtige Grundsätze:
- Sei streng aber fair; zweifelhafte Evidenz führt zu "teilkonform"
- Fehlende Evidenz führt zu "nicht_prüfbar", NICHT automatisch zu "nicht_konform"
- Zitiere immer die Quelle (Dateiname + Abschnitt) für deine Belege
- Formuliere Mängel im offiziellen BaFin-Stil: sachlich, präzise, ohne Schuldzuweisungen

Antworte AUSSCHLIESSLICH als valides JSON mit dieser Struktur:
{
  "bewertung": "konform|teilkonform|nicht_konform|nicht_prüfbar",
  "begruendung": "Ausführliche Begründung (3-8 Sätze)",
  "belegte_textstellen": ["Zitat 1 (Quelle: datei.pdf, S.X)", "..."],
  "mangel_text": "Formulierung im BaFin-Stil oder null wenn konform",
  "empfehlungen": ["Konkrete Maßnahme 1", "..."],
  "quellen": ["datei.pdf", "interview.json"]
}
"""


class GwGPrueferAgent:
    """
    Bewertet ein einzelnes Prüffeld durch:
    1. RAG-Retrieval relevanter Dokumentenstellen
    2. LLM-Bewertung gegen die Prüffrage
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        model: str = "claude-opus-4-5",
        top_k: int = 8,
        temperature: float = 0.1,
    ):
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        self.llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=2048)

    def pruefe_feld(self, prueffeld: dict) -> Befund:
        """Hauptmethode: Prüft ein einzelnes Prüffeld und gibt einen Befund zurück."""

        # 1. Relevante Evidenz aus dem Index holen
        evidenz_nodes = self._retrieve_evidence(prueffeld)
        evidenz_text = self._format_evidence(evidenz_nodes, prueffeld)

        # 2. LLM-Bewertung
        llm_result = self._evaluate_with_llm(prueffeld, evidenz_text)

        # 3. Befund zusammenbauen
        return Befund(
            prueffeld_id=prueffeld["id"],
            frage=prueffeld["frage"],
            bewertung=Bewertung(llm_result.get("bewertung", "nicht_prüfbar")),
            begruendung=llm_result.get("begruendung", ""),
            belegte_textstellen=llm_result.get("belegte_textstellen", []),
            empfehlungen=llm_result.get("empfehlungen", []),
            mangel_text=llm_result.get("mangel_text"),
            schweregrad=prueffeld.get("schweregrad"),
            quellen=llm_result.get("quellen", []),
        )

    def _retrieve_evidence(self, prueffeld: dict) -> list:
        """Baut eine optimierte Suchanfrage und holt relevante Chunks."""
        # Kombiniere Frage + erwartete Evidenz für besseres Retrieval
        search_query = f"""
        {prueffeld['frage']}
        Relevante Begriffe: {', '.join(prueffeld.get('erwartete_evidenz', []))}
        Rechtsgrundlage: {prueffeld.get('rechtsgrundlage', '')}
        """
        return self.retriever.retrieve(search_query.strip())

    def _format_evidence(self, nodes: list, prueffeld: dict) -> str:
        """Formatiert die Evidenz für den LLM-Prompt."""
        if not nodes:
            return "KEINE EVIDENZ GEFUNDEN – keine relevanten Dokumente im Prüfungskorpus."

        # Filtere nach input_typen wenn angegeben
        allowed_types = set(prueffeld.get("input_typen", []))

        lines = ["=== GEFUNDENE EVIDENZ ===\n"]
        for i, node in enumerate(nodes, 1):
            meta = node.metadata or {}
            source = meta.get("source", "unbekannt")
            input_type = meta.get("input_type", "unbekannt")

            # Type-Filter (soft – bei leerer Liste alles zulassen)
            if allowed_types and input_type not in allowed_types:
                continue

            score = getattr(node, "score", None)
            score_str = f" (Relevanz: {score:.2f})" if score else ""

            lines.append(f"--- Evidenz {i}: {source} [{input_type}]{score_str} ---")

            # Screenshots: gesondert behandeln
            if input_type == "screenshot":
                lines.append(f"[Screenshot-Datei: {source} – visuelle Prüfung erforderlich]")
            else:
                lines.append(node.get_content()[:1500])  # Token-Limit beachten
            lines.append("")

        return "\n".join(lines)

    def _evaluate_with_llm(self, prueffeld: dict, evidenz_text: str) -> dict:
        """Sendet Prüffrage + Evidenz an das LLM und parst das JSON-Ergebnis."""
        user_prompt = f"""
## PRÜFFELD: {prueffeld['id']}
**Frage:** {prueffeld['frage']}
**Rechtsgrundlage:** {', '.join(prueffeld.get('rechtsgrundlagen', [prueffeld.get('rechtsgrundlage', '')]))}
**Erwartete Evidenz:** {', '.join(prueffeld.get('erwartete_evidenz', []))}
**Schweregrad:** {prueffeld.get('schweregrad', 'unbekannt')}
**Bewertungskriterien:** {prueffeld.get('bewertungskriterien', '')}

{evidenz_text}

Bewerte dieses Prüffeld und antworte als JSON.
"""
        messages = [
            SystemMessage(content=SYSTEM_PROMPT_PRUEFER),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            raw = response.content.strip()
            # JSON extrahieren (manchmal mit ```json ... ``` umhüllt)
            if "```" in raw:
                raw = raw.split("```json")[-1].split("```")[0].strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            return {
                "bewertung": "nicht_prüfbar",
                "begruendung": f"LLM-Antwort konnte nicht geparst werden: {e}",
                "belegte_textstellen": [],
                "mangel_text": None,
                "empfehlungen": [],
                "quellen": [],
            }
        except Exception as e:
            return {
                "bewertung": "nicht_prüfbar",
                "begruendung": f"Fehler bei der Bewertung: {e}",
                "belegte_textstellen": [],
                "mangel_text": None,
                "empfehlungen": [],
                "quellen": [],
            }
