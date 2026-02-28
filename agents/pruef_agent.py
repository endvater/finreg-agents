"""
FinRegAgents – Prüfer-Agent (v2)
Ein Agent pro Prüffeld: RAG-Suche → Qualitäts-Gate → LLM-Bewertung → Validierung → Befund

Änderungen gegenüber v1:
  - Regulatorik-spezifischer System-Prompt (nicht mehr GwG-hardcoded)
  - Retrieval-Score-Threshold (Low-Quality → automatisch nicht_prüfbar)
  - Confidence-Score aus Retrieval-Score + Evidenz-Coverage + Type-Match
  - Strukturelle Validierung der LLM-Antwort (Quellen-Cross-Check, §-Validierung)
  - Robusteres JSON-Parsing mit Regex-Fallback
  - Einheitliches Framework (nur LangChain, kein LlamaIndex-LLM)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import json
import re
import hashlib

from llama_index.core import VectorStoreIndex
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
    NICHT_PRUEFBAR = "nicht_prüfbar"


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
    confidence: float = 0.0
    review_erforderlich: bool = False
    validierungshinweise: list[str] = field(default_factory=list)


@dataclass
class Sektionsergebnis:
    sektion_id: str
    titel: str
    befunde: list[Befund] = field(default_factory=list)

    @property
    def kritische_befunde(self) -> list[Befund]:
        return [b for b in self.befunde
                if b.bewertung in (Bewertung.NICHT_KONFORM, Bewertung.TEILKONFORM)]

    @property
    def review_quote(self) -> float:
        """Anteil der Befunde, die manuelles Review erfordern."""
        if not self.befunde:
            return 0.0
        return sum(1 for b in self.befunde if b.review_erforderlich) / len(self.befunde)


# ------------------------------------------------------------------ #
# Regulatorik-spezifische System-Prompts
# ------------------------------------------------------------------ #

SYSTEM_PROMPTS = {
    "gwg": """Du bist ein erfahrener Sonderprüfer der BaFin mit Spezialisierung auf Geldwäscheprävention.
Du führst eine Sonderprüfung gemäß §25h KWG und GwG durch.
Relevante Rechtsrahmen: GwG 2017 i.d.F. 2024, §25h KWG, BaFin AuA GwG, AMLA-Leitlinien.""",

    "dora": """Du bist ein erfahrener Prüfer mit Spezialisierung auf digitale operationale Resilienz.
Du führst eine Prüfung gemäß DORA (EU) 2022/2554 durch.
Relevante Rechtsrahmen: DORA Art. 5-46, RTS ICT Risk, RTS Incident Reporting, TIBER-EU.""",

    "marisk": """Du bist ein erfahrener Prüfer der BaFin mit Spezialisierung auf Risikomanagement.
Du führst eine Prüfung gemäß MaRisk (BaFin-Rundschreiben) und §25a KWG durch.
Relevante Rechtsrahmen: MaRisk 2023 (AT/BT), §25a KWG, EBA-Leitlinien.""",

    "wphg": """Du bist ein erfahrener Prüfer mit Spezialisierung auf Wertpapieraufsicht und Compliance.
Du führst eine Prüfung gemäß WpHG und MaComp durch.
Relevante Rechtsrahmen: WpHG, MaComp, MAR (EU) Nr. 596/2014, MiFID II.""",
}

SYSTEM_PROMPT_TEMPLATE = """{regulatorik_kontext}

Deine Aufgabe:
1. Analysiere die bereitgestellten Dokumentenausschnitte (Evidenz)
2. Beantworte die Prüffrage präzise und mit direktem Bezug auf die Evidenz
3. Bewerte gemäß: konform | teilkonform | nicht_konform | nicht_prüfbar
4. Belege deine Bewertung mit konkreten Textstellen aus den Dokumenten
5. Formuliere ggf. einen Mangel im BaFin-Stil und konkrete Empfehlungen
6. Schätze deine eigene Sicherheit ein (confidence_self: 0.0 bis 1.0)

Wichtige Grundsätze:
- Sei streng aber fair; zweifelhafte Evidenz führt zu "teilkonform"
- Fehlende Evidenz führt zu "nicht_prüfbar", NICHT automatisch zu "nicht_konform"
- Zitiere immer die Quelle (Dateiname + Abschnitt) für deine Belege
- Formuliere Mängel sachlich, präzise, ohne Schuldzuweisungen
- Zitiere NUR Quellen, die tatsächlich in der bereitgestellten Evidenz vorkommen
- Wenn die Evidenz nicht ausreicht, bewerte als "nicht_prüfbar" – erfinde keine Belege

Antworte AUSSCHLIESSLICH als valides JSON mit dieser Struktur:
{{
  "bewertung": "konform|teilkonform|nicht_konform|nicht_prüfbar",
  "begruendung": "Ausführliche Begründung (3-8 Sätze)",
  "belegte_textstellen": ["Zitat 1 (Quelle: datei.pdf, S.X)", "..."],
  "mangel_text": "Formulierung im BaFin-Stil oder null wenn konform",
  "empfehlungen": ["Konkrete Maßnahme 1", "..."],
  "quellen": ["datei.pdf", "interview.json"],
  "confidence_self": 0.85
}}
"""


# ------------------------------------------------------------------ #
# Confidence-Berechnung
# ------------------------------------------------------------------ #

# Thresholds
RETRIEVAL_SCORE_MIN = 0.35          # Unter diesem Wert: nicht_prüfbar
CONFIDENCE_AUTO_REJECT = 0.4        # Unter diesem Wert: automatisch nicht_prüfbar
CONFIDENCE_REVIEW_THRESHOLD = 0.7   # Unter diesem Wert: manuelles Review markieren
SEKTION_REVIEW_ESCALATION = 0.3     # Ab diesem Anteil: Sektion eskalieren


def compute_confidence(
    retrieval_scores: list[float],
    erwartete_evidenz: list[str],
    gefundene_quellen: list[str],
    erlaubte_typen: set[str],
    gefundene_typen: set[str],
    llm_confidence: float,
) -> float:
    """
    Berechnet einen kombinierten Confidence-Score aus vier Signalen:
      - Retrieval-Score (30%): Durchschnittliche Relevanz der top-k Chunks
      - Evidenz-Coverage (30%): Anteil der erwarteten Evidenz die gefunden wurde
      - Type-Match (20%): Stimmen die Dokumenttypen überein?
      - LLM-Self-Assessment (20%): Selbsteinschätzung des Modells
    """
    # 1. Retrieval-Score (Durchschnitt der Top-Scores, normalisiert auf 0-1)
    if retrieval_scores:
        avg_score = sum(retrieval_scores) / len(retrieval_scores)
        retrieval_signal = min(max(avg_score, 0.0), 1.0)
    else:
        retrieval_signal = 0.0

    # 2. Evidenz-Coverage: Wie viele der erwarteten Begriffe tauchen in Quellen auf?
    if erwartete_evidenz:
        matched = 0
        quellen_lower = " ".join(gefundene_quellen).lower()
        for ev in erwartete_evidenz:
            # Fuzzy: Teilwort-Match in Quelldateinamen
            if any(tok in quellen_lower for tok in ev.lower().split()):
                matched += 1
        coverage_signal = matched / len(erwartete_evidenz)
    else:
        coverage_signal = 0.5  # Neutral wenn nichts erwartet

    # 3. Type-Match
    if erlaubte_typen:
        overlap = erlaubte_typen & gefundene_typen
        type_signal = len(overlap) / len(erlaubte_typen) if erlaubte_typen else 0.5
    else:
        type_signal = 1.0  # Keine Einschränkung = OK

    # 4. LLM-Self-Assessment (begrenzt auf 0-1)
    llm_signal = min(max(llm_confidence, 0.0), 1.0)

    # Gewichteter Score
    confidence = (
        0.30 * retrieval_signal
        + 0.30 * coverage_signal
        + 0.20 * type_signal
        + 0.20 * llm_signal
    )
    return round(confidence, 3)


# ------------------------------------------------------------------ #
# Strukturelle Validierung
# ------------------------------------------------------------------ #

# Bekannte Paragraphen-Muster je Regulatorik
KNOWN_LAW_PATTERNS = {
    "gwg": re.compile(r"§\s*\d+[a-z]?\s*(Abs\.\s*\d+)?\s*(GwG|KWG)"),
    "dora": re.compile(r"Art\.\s*\d+\s*(Abs\.\s*\d+)?\s*DORA"),
    "marisk": re.compile(r"(MaRisk\s*(AT|BT)\s*\d+(\.\d+)*|§\s*\d+[a-z]?\s*(Abs\.\s*\d+)?\s*KWG)"),
    "wphg": re.compile(r"(§\s*\d+[a-z]?\s*(Abs\.\s*\d+)?\s*(WpHG|MaComp)|Art\.\s*\d+\s*MA[RD])"),
}


def validate_befund_structure(
    llm_result: dict,
    retrieved_sources: set[str],
    regulatorik: str,
) -> list[str]:
    """
    Strukturelle Validierung des LLM-Outputs. Gibt eine Liste von Warnungen zurück.
    """
    warnings = []

    # 1. Quellen-Cross-Check: Zitiert der Agent Quellen, die nicht retrieved wurden?
    agent_quellen = set(llm_result.get("quellen", []))
    phantom_quellen = agent_quellen - retrieved_sources
    if phantom_quellen:
        warnings.append(
            f"Phantom-Quellen (nicht im Retrieval): {', '.join(phantom_quellen)}"
        )

    # 2. Platzhalter-Check in Begründung und Mangel
    for field_name in ("begruendung", "mangel_text"):
        text = llm_result.get(field_name) or ""
        if re.search(r"\{[^}]*\}", text):
            warnings.append(f"Unaufgelöster Platzhalter in '{field_name}'")

    # 3. Bewertung ≠ nicht_prüfbar aber keine Textstellen
    bewertung = llm_result.get("bewertung", "")
    textstellen = llm_result.get("belegte_textstellen", [])
    if bewertung in ("konform", "teilkonform", "nicht_konform") and not textstellen:
        warnings.append(
            f"Bewertung '{bewertung}' ohne belegte Textstellen"
        )

    # 4. Mangel_text vorhanden bei konform
    if bewertung == "konform" and llm_result.get("mangel_text"):
        warnings.append("Mangel-Text bei 'konform'-Bewertung")

    # 5. Mangel_text fehlt bei nicht_konform
    if bewertung == "nicht_konform" and not llm_result.get("mangel_text"):
        warnings.append("Kein Mangel-Text bei 'nicht_konform'-Bewertung")

    return warnings


# ------------------------------------------------------------------ #
# JSON-Extraktion (robust)
# ------------------------------------------------------------------ #

_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def extract_json(raw: str) -> dict:
    """Extrahiert JSON aus einer LLM-Antwort – robust gegen Markdown-Wrapping."""
    raw = raw.strip()

    # Versuch 1: Direkt parsen
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Versuch 2: ```json ... ``` Block
    m = _JSON_BLOCK_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # Versuch 3: Erstes JSON-Objekt finden
    m = _JSON_OBJECT_RE.search(raw)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("Kein gültiges JSON in LLM-Antwort gefunden", raw, 0)


# ------------------------------------------------------------------ #
# Prüfer-Agent
# ------------------------------------------------------------------ #

class PrueferAgent:
    """
    Bewertet ein einzelnes Prüffeld durch:
    1. RAG-Retrieval relevanter Dokumentenstellen
    2. Retrieval-Quality-Gate
    3. LLM-Bewertung gegen die Prüffrage
    4. Strukturelle Validierung
    5. Confidence-Scoring
    """

    def __init__(
        self,
        index: VectorStoreIndex,
        regulatorik: str = "gwg",
        model: str = "claude-sonnet-4-5-20250514",
        top_k: int = 8,
        temperature: float = 0.1,
        retrieval_score_min: float = RETRIEVAL_SCORE_MIN,
    ):
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        self.llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=2048)
        self.regulatorik = regulatorik
        self.retrieval_score_min = retrieval_score_min

        # System-Prompt für die gewählte Regulatorik
        kontext = SYSTEM_PROMPTS.get(regulatorik, SYSTEM_PROMPTS["gwg"])
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(regulatorik_kontext=kontext)

    def pruefe_feld(self, prueffeld: dict) -> Befund:
        """Hauptmethode: Prüft ein einzelnes Prüffeld und gibt einen Befund zurück."""

        # 1. Relevante Evidenz aus dem Index holen
        evidenz_nodes = self._retrieve_evidence(prueffeld)

        # 2. Retrieval-Quality-Gate
        scores = [getattr(n, "score", 0.0) or 0.0 for n in evidenz_nodes]
        good_nodes = [n for n, s in zip(evidenz_nodes, scores) if s >= self.retrieval_score_min]

        if not good_nodes:
            avg_score = sum(scores) / len(scores) if scores else 0.0
            return Befund(
                prueffeld_id=prueffeld["id"],
                frage=prueffeld["frage"],
                bewertung=Bewertung.NICHT_PRUEFBAR,
                begruendung=(
                    f"Retrieval-Quality-Gate: Keine Dokumente mit ausreichender Relevanz "
                    f"gefunden (bester Score: {max(scores) if scores else 0:.2f}, "
                    f"Threshold: {self.retrieval_score_min:.2f}). "
                    f"Die verfügbaren Dokumente im Prüfungskorpus enthalten keine "
                    f"ausreichend relevanten Informationen zu diesem Prüffeld."
                ),
                schweregrad=prueffeld.get("schweregrad"),
                confidence=0.0,
                review_erforderlich=True,
                validierungshinweise=["Automatisch nicht_prüfbar: Retrieval-Score unter Threshold"],
            )

        # 3. Evidenz formatieren (nur gute Nodes)
        evidenz_text = self._format_evidence(good_nodes, prueffeld)
        retrieved_sources = {
            (n.metadata or {}).get("source", "unbekannt") for n in good_nodes
        }
        retrieved_types = {
            (n.metadata or {}).get("input_type", "unbekannt") for n in good_nodes
        }

        # 4. LLM-Bewertung
        llm_result = self._evaluate_with_llm(prueffeld, evidenz_text)

        # 5. Strukturelle Validierung
        val_warnings = validate_befund_structure(
            llm_result, retrieved_sources, self.regulatorik
        )

        # 6. Confidence-Score berechnen
        good_scores = [s for s in scores if s >= self.retrieval_score_min]
        confidence = compute_confidence(
            retrieval_scores=good_scores,
            erwartete_evidenz=prueffeld.get("erwartete_evidenz", []),
            gefundene_quellen=list(retrieved_sources),
            erlaubte_typen=set(prueffeld.get("input_typen", [])),
            gefundene_typen=retrieved_types,
            llm_confidence=llm_result.get("confidence_self", 0.5),
        )

        # 7. Confidence-basierte Entscheidung
        bewertung_str = llm_result.get("bewertung", "nicht_prüfbar")
        review_erforderlich = False

        if confidence < CONFIDENCE_AUTO_REJECT:
            bewertung_str = "nicht_prüfbar"
            val_warnings.append(
                f"Confidence zu niedrig ({confidence:.2f} < {CONFIDENCE_AUTO_REJECT}): "
                f"automatisch auf nicht_prüfbar gesetzt"
            )
            review_erforderlich = True
        elif confidence < CONFIDENCE_REVIEW_THRESHOLD:
            review_erforderlich = True
            val_warnings.append(
                f"Review erforderlich: Confidence {confidence:.2f} < {CONFIDENCE_REVIEW_THRESHOLD}"
            )

        # Wenn Validierungswarnungen vorliegen → Review erzwingen
        if val_warnings:
            review_erforderlich = True

        return Befund(
            prueffeld_id=prueffeld["id"],
            frage=prueffeld["frage"],
            bewertung=Bewertung(bewertung_str),
            begruendung=llm_result.get("begruendung", ""),
            belegte_textstellen=llm_result.get("belegte_textstellen", []),
            empfehlungen=llm_result.get("empfehlungen", []),
            mangel_text=llm_result.get("mangel_text"),
            schweregrad=prueffeld.get("schweregrad"),
            quellen=llm_result.get("quellen", []),
            confidence=confidence,
            review_erforderlich=review_erforderlich,
            validierungshinweise=val_warnings,
        )

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def _retrieve_evidence(self, prueffeld: dict) -> list:
        """Baut eine optimierte Suchanfrage und holt relevante Chunks."""
        parts = [prueffeld["frage"]]
        if prueffeld.get("erwartete_evidenz"):
            parts.append(f"Relevante Begriffe: {', '.join(prueffeld['erwartete_evidenz'])}")
        rg = prueffeld.get("rechtsgrundlagen", [])
        if isinstance(rg, list) and rg:
            parts.append(f"Rechtsgrundlage: {', '.join(rg)}")
        elif isinstance(rg, str) and rg:
            parts.append(f"Rechtsgrundlage: {rg}")
        return self.retriever.retrieve("\n".join(parts))

    def _format_evidence(self, nodes: list, prueffeld: dict) -> str:
        """Formatiert die Evidenz für den LLM-Prompt."""
        if not nodes:
            return "KEINE EVIDENZ GEFUNDEN – keine relevanten Dokumente im Prüfungskorpus."

        allowed_types = set(prueffeld.get("input_typen", []))
        lines = ["=== GEFUNDENE EVIDENZ ===\n"]

        for i, node in enumerate(nodes, 1):
            meta = node.metadata or {}
            source = meta.get("source", "unbekannt")
            input_type = meta.get("input_type", "unbekannt")

            # Soft-Filter: Bei leerer Liste alles zulassen
            if allowed_types and input_type not in allowed_types:
                continue

            score = getattr(node, "score", None)
            score_str = f" (Relevanz: {score:.2f})" if score else ""

            lines.append(f"--- Evidenz {i}: {source} [{input_type}]{score_str} ---")

            if input_type == "screenshot":
                lines.append(f"[Screenshot-Datei: {source} – visuelle Prüfung durch Mensch erforderlich]")
            else:
                lines.append(node.get_content()[:2000])
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # LLM-Bewertung
    # ------------------------------------------------------------------ #

    def _evaluate_with_llm(self, prueffeld: dict, evidenz_text: str) -> dict:
        """Sendet Prüffrage + Evidenz an das LLM und parst das JSON-Ergebnis."""
        rg = prueffeld.get("rechtsgrundlagen", [])
        if isinstance(rg, list):
            rg_str = ", ".join(rg)
        else:
            rg_str = str(rg)

        user_prompt = f"""## PRÜFFELD: {prueffeld['id']}
**Frage:** {prueffeld['frage']}
**Rechtsgrundlage:** {rg_str}
**Erwartete Evidenz:** {', '.join(prueffeld.get('erwartete_evidenz', []))}
**Schweregrad:** {prueffeld.get('schweregrad', 'unbekannt')}
**Bewertungskriterien:** {prueffeld.get('bewertungskriterien', '')}

{evidenz_text}

Bewerte dieses Prüffeld und antworte als JSON.
"""
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return extract_json(response.content)
        except json.JSONDecodeError as e:
            return {
                "bewertung": "nicht_prüfbar",
                "begruendung": f"LLM-Antwort konnte nicht als JSON geparst werden: {e}",
                "belegte_textstellen": [],
                "mangel_text": None,
                "empfehlungen": [],
                "quellen": [],
                "confidence_self": 0.0,
            }
        except Exception as e:
            return {
                "bewertung": "nicht_prüfbar",
                "begruendung": f"Fehler bei der LLM-Bewertung: {type(e).__name__}: {e}",
                "belegte_textstellen": [],
                "mangel_text": None,
                "empfehlungen": [],
                "quellen": [],
                "confidence_self": 0.0,
            }


# Rückwärtskompatibilität
GwGPrueferAgent = PrueferAgent
SektionsergebniS = Sektionsergebnis  # Typo-Alias
