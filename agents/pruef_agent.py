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
import logging
import re
import time
from collections import Counter

from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from agents.llm_factory import build_llm
from agents.provenance import ClaimProvenance, annotate_claims
from agents.term_checker import TermDriftChecker
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Datenmodelle
# ------------------------------------------------------------------ #


class Bewertung(str, Enum):
    KONFORM = "konform"
    TEILKONFORM = "teilkonform"
    NICHT_KONFORM = "nicht_konform"
    NICHT_PRUEFBAR = "nicht_prüfbar"
    DISPUTED = "disputed"


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
    confidence_level: str = "low"
    confidence_guards: dict = field(default_factory=dict)
    low_confidence_reasons: list[str] = field(default_factory=list)
    token_usage: dict = field(default_factory=dict)
    claim_list: list[dict] = field(default_factory=list)
    review_erforderlich: bool = False
    validierungshinweise: list[str] = field(default_factory=list)
    term_drift_warnings: list[str] = field(default_factory=list)
    claim_provenance: list[ClaimProvenance] = field(default_factory=list)
    disputed_positions: Optional[dict] = None


@dataclass
class Sektionsergebnis:
    sektion_id: str
    titel: str
    befunde: list[Befund] = field(default_factory=list)

    @property
    def kritische_befunde(self) -> list[Befund]:
        return [
            b
            for b in self.befunde
            if b.bewertung in (Bewertung.NICHT_KONFORM, Bewertung.TEILKONFORM)
        ]

    @property
    def review_quote(self) -> float:
        """Anteil der Befunde, die manuelles Review erfordern."""
        if not self.befunde:
            return 0.0
        return sum(1 for b in self.befunde if b.review_erforderlich) / len(self.befunde)


@dataclass
class AdversarialErgebnis:
    """Ergebnis des adversarialen Prüf-Passes."""

    prueffeld_id: str
    adversarial_bewertung: Bewertung
    schwachstellen: list[str] = field(default_factory=list)
    fehlende_nachweise: list[str] = field(default_factory=list)


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
# Adversarial Prompting Layer
# ------------------------------------------------------------------ #

ADVERSARIAL_SYSTEM_PROMPTS = {
    "gwg": """Du bist ein kritischer Gutachter, der Schwachstellen in Geldwäscheprüfungen aufdeckt.
Deine Aufgabe: Finde alle Gründe, warum die geprüfte Compliance-Anforderung NICHT oder nur unzureichend erfüllt sein könnte.
Prüfrahmen: GwG 2017 i.d.F. 2024, §25h KWG, BaFin AuA GwG, AMLA-Leitlinien.""",
    "dora": """Du bist ein kritischer Gutachter, der Schwachstellen in DORA-Prüfungen aufdeckt.
Deine Aufgabe: Finde alle Gründe, warum die geprüfte Resilienz-Anforderung NICHT oder nur unzureichend erfüllt sein könnte.
Prüfrahmen: DORA (EU) 2022/2554, RTS ICT Risk, RTS Incident Reporting, TIBER-EU.""",
    "marisk": """Du bist ein kritischer Gutachter, der Schwachstellen in MaRisk-Prüfungen aufdeckt.
Deine Aufgabe: Finde alle Gründe, warum die geprüfte Risikomanagement-Anforderung NICHT oder nur unzureichend erfüllt sein könnte.
Prüfrahmen: MaRisk 2023 (AT/BT), §25a KWG, EBA-Leitlinien.""",
    "wphg": """Du bist ein kritischer Gutachter, der Schwachstellen in WpHG/MaComp-Prüfungen aufdeckt.
Deine Aufgabe: Finde alle Gründe, warum die geprüfte Compliance-Anforderung NICHT oder nur unzureichend erfüllt sein könnte.
Prüfrahmen: WpHG, MaComp, MAR (EU) Nr. 596/2014, MiFID II.""",
}

ADVERSARIAL_PROMPT_TEMPLATE = """{regulatorik_kontext}

Deine Aufgabe (adversarial):
1. Analysiere dieselben Dokumentenausschnitte wie ein normaler Prüfer
2. Beantworte die Prüffrage – aber aus der Perspektive eines kritischen Gegenspielers
3. Bewerte gemäß: konform | teilkonform | nicht_konform | nicht_prüfbar
4. Identifiziere aktiv Schwachstellen und Gegenargumente
5. Benenne fehlende oder unzureichende Nachweise konkret
6. Schätze deine Sicherheit ein (confidence_self: 0.0 bis 1.0)

Wichtige Grundsätze (adversarial):
- Bevorzuge strengere Bewertungen bei Ambiguität
- Formale Dokumente ohne Prozessnachweis gelten als unzureichend
- Fehlende Audit-Trails, Schulungsnachweise oder Tests sind Mängel
- Zitiere NUR Quellen aus der bereitgestellten Evidenz – erfinde keine

Antworte AUSSCHLIESSLICH als valides JSON mit dieser Struktur:
{{
  "bewertung": "konform|teilkonform|nicht_konform|nicht_prüfbar",
  "begruendung": "Kritische Würdigung der Evidenz (3-5 Sätze)",
  "schwachstellen": ["Schwachstelle 1", "Schwachstelle 2"],
  "fehlende_nachweise": ["Fehlender Nachweis 1", "..."],
  "quellen": ["datei.pdf"],
  "confidence_self": 0.75
}}
"""


# ------------------------------------------------------------------ #
# Confidence-Berechnung
# ------------------------------------------------------------------ #

# Thresholds
RETRIEVAL_SCORE_MIN = 0.35  # Unter diesem Wert: nicht_prüfbar
CONFIDENCE_AUTO_REJECT = 0.4  # Unter diesem Wert: automatisch nicht_prüfbar
CONFIDENCE_REVIEW_THRESHOLD = 0.7  # Unter diesem Wert: manuelles Review markieren
SEKTION_REVIEW_ESCALATION = 0.3  # Ab diesem Anteil: Sektion eskalieren
CONFIDENCE_HIGH_THRESHOLD = 0.8
CONFIDENCE_MEDIUM_THRESHOLD = 0.5

# Confidence Guards (Issue #28)
MIN_INPUT_TOKENS = 300
MIN_DISTINCT_SOURCES = 2
MIN_EVIDENCE_QUOTES = 1

# Retry-Konfiguration für API-Fehler
LLM_MAX_RETRIES = 3
LLM_RETRY_BASE_DELAY = 2.0  # Sekunden, verdoppelt sich je Versuch

# Regex für präzises Token-Splitting bei Dateinamen (Fuzzy-Matching-Fix)
_FILENAME_SEP_RE = re.compile(r"[\s._\-/]+")


def estimate_tokens(text: str) -> int:
    """Pragmatische Token-Schätzung (~4 Zeichen pro Token)."""
    if not text:
        return 0
    return max(1, len(text) // 4)


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
        # Dateinamen auf Separator-Grenzen aufteilen, um False Positives zu vermeiden
        # z.B. "log" soll NICHT "dialog.pdf" matchen
        quellen_tokens: set[str] = set()
        for q in gefundene_quellen:
            quellen_tokens.update(t for t in _FILENAME_SEP_RE.split(q.lower()) if t)
        for ev in erwartete_evidenz:
            ev_tokens = set(ev.lower().split())
            if ev_tokens & quellen_tokens:
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


def confidence_level_from_score(score: float) -> str:
    if score >= CONFIDENCE_HIGH_THRESHOLD:
        return "high"
    if score >= CONFIDENCE_MEDIUM_THRESHOLD:
        return "medium"
    return "low"


def evaluate_confidence_guards(
    input_tokens: int,
    distinct_sources: int,
    evidence_quotes: int,
) -> dict:
    violations = []
    if input_tokens < MIN_INPUT_TOKENS:
        violations.append("MIN_INPUT_TOKENS")
    if distinct_sources < MIN_DISTINCT_SOURCES:
        violations.append("MIN_DISTINCT_SOURCES")
    if evidence_quotes < MIN_EVIDENCE_QUOTES:
        violations.append("MIN_EVIDENCE_QUOTES")

    return {
        "passed": not violations,
        "violations": violations,
        "metrics": {
            "input_tokens": input_tokens,
            "distinct_sources": distinct_sources,
            "evidence_quotes": evidence_quotes,
        },
    }


# ------------------------------------------------------------------ #
# Strukturelle Validierung
# ------------------------------------------------------------------ #

# Bekannte Paragraphen-Muster je Regulatorik
KNOWN_LAW_PATTERNS = {
    "gwg": re.compile(r"§\s*\d+[a-z]?\s*(Abs\.\s*\d+)?\s*(GwG|KWG)"),
    "dora": re.compile(r"Art\.\s*\d+\s*(Abs\.\s*\d+)?\s*DORA"),
    "marisk": re.compile(
        r"(MaRisk\s*(AT|BT)\s*\d+(\.\d+)*|§\s*\d+[a-z]?\s*(Abs\.\s*\d+)?\s*KWG)"
    ),
    "wphg": re.compile(
        r"(§\s*\d+[a-z]?\s*(Abs\.\s*\d+)?\s*(WpHG|MaComp)|Art\.\s*\d+\s*MA[RD])"
    ),
}
# Erfasst generische Normzitate inkl. "Abs." (Punkt muss erlaubt sein),
# begrenzt aber die Match-Länge durch Tokenanzahl.
GENERIC_LAW_REF_RE = re.compile(
    r"(§\s*\d+[a-z]?(?:\s+\S+){0,6}|Art\.\s*\d+[a-z]?(?:\s+\S+){0,6})"
)
REG_GUARDRAIL_RE = re.compile(r"(§|Art\.|MaRisk|DORA|GwG|KWG|WpHG|MaComp)")

MARKETING_TERMS = (
    "kundenfokus",
    "innovativ",
    "premium",
    "marktführ",
    "vision",
    "mission",
    "growth",
    "brand",
    "campaign",
    "marketing",
)
CONTROL_TERMS = (
    "kontrolle",
    "prozess",
    "freigabe",
    "monitoring",
    "review",
    "dokument",
    "nachweis",
    "risiko",
    "policy",
    "verfahren",
    "pruefung",
    "prüfung",
    "audit",
    "eskalation",
)
NON_CONTROL_TERMS = (
    "presse",
    "blog",
    "karriere",
    "event",
    "newsletter",
    "werbung",
)

# Numerischer Schweregrad je Bewertung (für Divergenzberechnung)
BEWERTUNG_SEVERITY = {
    "konform": 0,
    "teilkonform": 1,
    "nicht_konform": 2,
    "nicht_prüfbar": 3,
    "disputed": 3,
}

# Confidence-Penalty je Divergenzstufe
_ADVERSARIAL_PENALTY = {1: 0.05, 2: 0.15, 3: 0.20}
NORM_REF_RE = re.compile(
    r"(§\s*\d+[a-z]?(?:\s*Abs\.\s*\d+)?(?:\s*(?:GwG|KWG|WpHG|MaComp))?"
    r"|Art\.\s*\d+[a-z]?(?:\s*Abs\.\s*\d+)?(?:\s*(?:DORA|MAR))?)"
)


def _extract_norm_refs(text: str) -> set[str]:
    refs = set()
    for match in NORM_REF_RE.finditer(text or ""):
        refs.add(re.sub(r"\s+", " ", match.group(0).strip()))
    return refs


def validate_befund_structure(
    llm_result: dict,
    retrieved_sources: set[str],
    regulatorik: str,
    evidence_text: str = "",
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
        warnings.append(f"Bewertung '{bewertung}' ohne belegte Textstellen")

    # 4. Mangel_text vorhanden bei konform
    if bewertung == "konform" and llm_result.get("mangel_text"):
        warnings.append("Mangel-Text bei 'konform'-Bewertung")

    # 5. Mangel_text fehlt bei nicht_konform
    if bewertung == "nicht_konform" and not llm_result.get("mangel_text"):
        warnings.append("Kein Mangel-Text bei 'nicht_konform'-Bewertung")

    # 6. Rechtszitate auf regulatorik-spezifische Plausibilität prüfen
    law_pattern = KNOWN_LAW_PATTERNS.get(regulatorik)
    if law_pattern:
        prüftexte = [
            llm_result.get("begruendung", "") or "",
            llm_result.get("mangel_text", "") or "",
            *[
                str(t)
                for t in (llm_result.get("belegte_textstellen") or [])
                if t is not None
            ],
        ]
        suspicious_refs = set()
        for text in prüftexte:
            for ref in GENERIC_LAW_REF_RE.findall(text):
                ref_clean = ref.strip()
                if ref_clean and not law_pattern.search(ref_clean):
                    suspicious_refs.add(ref_clean)
        if suspicious_refs:
            refs = ", ".join(sorted(suspicious_refs))
            warnings.append(f"Unplausible Rechtszitate für '{regulatorik}': {refs}")

    # 7. Context-Drift: zentrale Normreferenzen aus Evidenz sollen erhalten bleiben
    if evidence_text:
        refs_evidence = _extract_norm_refs(evidence_text)
        refs_befund = _extract_norm_refs(
            " ".join(
                [
                    llm_result.get("begruendung", "") or "",
                    llm_result.get("mangel_text", "") or "",
                    " ".join(llm_result.get("belegte_textstellen", []) or []),
                ]
            )
        )
        missing_refs = refs_evidence - refs_befund
        if missing_refs:
            sample = ", ".join(sorted(missing_refs)[:3])
            warnings.append(
                "Context-Drift-Verdacht: Normreferenzen aus Evidenz fehlen im "
                f"Befund ({sample})"
            )

    return warnings


def build_claim_annotations(
    llm_result: dict,
    nodes: list,
    retrieved_sources: set[str],
) -> list[dict]:
    """
    Leitet eine auditierbare Claim-Liste aus Textstellen und Retrieval-Metadaten ab.
    """
    raw_claims = [str(t).strip() for t in (llm_result.get("belegte_textstellen") or [])]
    raw_claims = [c for c in raw_claims if c]
    if not raw_claims:
        begruendung = (llm_result.get("begruendung") or "").strip()
        if begruendung:
            raw_claims = [begruendung]
    if not raw_claims:
        return []

    provenance = []
    seen = set()
    for node in nodes:
        md = getattr(node, "metadata", {}) or {}
        signature = (
            str(md.get("source", "unbekannt")),
            str(md.get("chunk_id", "")),
            str(md.get("page_label", md.get("start_char_idx", ""))),
        )
        if signature in seen:
            continue
        seen.add(signature)
        provenance.append(
            {
                "id": f"P{len(provenance) + 1}",
                "source": signature[0],
                "chunk_id": signature[1] or None,
                "position": signature[2] or None,
            }
        )

    source_count = len(retrieved_sources)
    status = (
        "corroborated"
        if source_count >= 2
        else "single-sourced"
        if source_count == 1
        else "unverified"
    )
    provenance_ids = [p["id"] for p in provenance]
    return [
        {
            "claim_id": f"C{idx + 1}",
            "text": claim,
            "status": status,
            "provenance_ids": provenance_ids,
            "provenance": provenance,
            "skeptiker_tag": "none",
        }
        for idx, claim in enumerate(raw_claims)
    ]


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
        provider: str = "anthropic",
        model: str | None = None,
        top_k: int = 8,
        temperature: float = 0.1,
        retrieval_score_min: float = RETRIEVAL_SCORE_MIN,
        adversarial: bool = False,
        evidence_relevance_filter: bool = False,
    ):
        self.retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        self.llm = build_llm(provider=provider, model=model, temperature=temperature)
        self.regulatorik = regulatorik
        self.retrieval_score_min = retrieval_score_min
        self.adversarial = adversarial
        self.evidence_relevance_filter = evidence_relevance_filter
        self.relevance_filter_stats = Counter()
        self.relevance_filter_drops = []

        # System-Prompt für die gewählte Regulatorik
        kontext = SYSTEM_PROMPTS.get(regulatorik, SYSTEM_PROMPTS["gwg"])
        self.system_prompt = SYSTEM_PROMPT_TEMPLATE.format(regulatorik_kontext=kontext)
        if self.adversarial:
            adv_kontext = ADVERSARIAL_SYSTEM_PROMPTS.get(
                regulatorik, ADVERSARIAL_SYSTEM_PROMPTS["gwg"]
            )
            self.adversarial_system_prompt = ADVERSARIAL_PROMPT_TEMPLATE.format(
                regulatorik_kontext=adv_kontext
            )

    def pruefe_feld(self, prueffeld: dict) -> Befund:
        """Hauptmethode: Prüft ein einzelnes Prüffeld und gibt einen Befund zurück."""

        # 1. Relevante Evidenz aus dem Index holen
        evidenz_nodes = self._retrieve_evidence(prueffeld)
        allowed_types = set(prueffeld.get("input_typen", []))
        scoped_nodes = [
            n
            for n in evidenz_nodes
            if not allowed_types
            or (n.metadata or {}).get("input_type", "unbekannt") in allowed_types
        ]
        if getattr(self, "evidence_relevance_filter", False):
            scoped_nodes, dropped = self._apply_relevance_filter(
                scoped_nodes, prueffeld
            )
            self.relevance_filter_stats["kept"] += len(scoped_nodes)
            self.relevance_filter_stats["dropped"] += len(dropped)
            for d in dropped:
                self.relevance_filter_stats[f"reason:{d['reason']}"] += 1
            self.relevance_filter_drops.extend(dropped)

        if not scoped_nodes:
            verfügbare_typen = sorted(
                {
                    (n.metadata or {}).get("input_type", "unbekannt")
                    for n in evidenz_nodes
                }
            )
            return Befund(
                prueffeld_id=prueffeld["id"],
                frage=prueffeld["frage"],
                bewertung=Bewertung.NICHT_PRUEFBAR,
                begruendung=(
                    "Keine Evidenz in erlaubten Dokumenttypen gefunden "
                    f"(erlaubt: {sorted(allowed_types) if allowed_types else 'alle'}, "
                    f"gefunden: {verfügbare_typen or ['keine']})."
                ),
                schweregrad=prueffeld.get("schweregrad"),
                confidence=0.0,
                confidence_level="low",
                confidence_guards=evaluate_confidence_guards(0, 0, 0),
                low_confidence_reasons=["NO_ALLOWED_EVIDENCE"],
                token_usage={"input": 0, "output": 0, "total": 0},
                review_erforderlich=True,
                validierungshinweise=[
                    "Automatisch nicht_prüfbar: Nur unzulässige Dokumenttypen im Retrieval"
                ],
            )

        # 2. Retrieval-Quality-Gate
        scores = [getattr(n, "score", 0.0) or 0.0 for n in scoped_nodes]
        good_nodes = [
            n for n, s in zip(scoped_nodes, scores) if s >= self.retrieval_score_min
        ]

        if not good_nodes:
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
                confidence_level="low",
                confidence_guards=evaluate_confidence_guards(0, 0, 0),
                low_confidence_reasons=["LOW_RETRIEVAL_QUALITY"],
                token_usage={"input": 0, "output": 0, "total": 0},
                review_erforderlich=True,
                validierungshinweise=[
                    "Automatisch nicht_prüfbar: Retrieval-Score unter Threshold"
                ],
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
        token_usage = llm_result.pop(
            "_token_usage", {"input": 0, "output": 0, "total": 0}
        )
        adv_ergebnis: Optional[AdversarialErgebnis] = None
        if self.adversarial:
            adv_ergebnis = self._adversarial_evaluate(prueffeld, evidenz_text)

        # 5. Strukturelle Validierung
        val_warnings = validate_befund_structure(
            llm_result,
            retrieved_sources,
            self.regulatorik,
            evidence_text=evidenz_text,
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
        confidence_level = confidence_level_from_score(confidence)

        # 6b. Confidence Guards
        guard_result = evaluate_confidence_guards(
            input_tokens=token_usage.get("input", 0),
            distinct_sources=len(retrieved_sources),
            evidence_quotes=len(llm_result.get("belegte_textstellen", []) or []),
        )
        low_confidence_reasons = list(guard_result["violations"])

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
            low_confidence_reasons.append("CONFIDENCE_AUTO_REJECT")
        elif confidence < CONFIDENCE_REVIEW_THRESHOLD:
            review_erforderlich = True
            val_warnings.append(
                f"Review erforderlich: Confidence {confidence:.2f} < {CONFIDENCE_REVIEW_THRESHOLD}"
            )
            low_confidence_reasons.append("CONFIDENCE_REVIEW_THRESHOLD")

        if not guard_result["passed"]:
            review_erforderlich = True
            val_warnings.append(
                f"Confidence-Guards verletzt: {', '.join(guard_result['violations'])}"
            )
            # Guards können confidence_level nur auf max. "medium" deckeln,
            # nicht weiter absenken ("low" bleibt "low")
            if confidence_level == "high":
                confidence_level = "medium"

        # Wenn Validierungswarnungen vorliegen → Review erzwingen
        if val_warnings:
            review_erforderlich = True
        low_confidence_reasons = list(dict.fromkeys(low_confidence_reasons))

        # 8. Term Drift Check
        befund_text_for_drift = " ".join(
            filter(
                None,
                [
                    llm_result.get("begruendung", "") or "",
                    llm_result.get("mangel_text", "") or "",
                    *[
                        str(t)
                        for t in (llm_result.get("belegte_textstellen") or [])
                        if t is not None
                    ],
                ],
            )
        )
        retrieved_chunk_texts = [n.get_content() for n in good_nodes]
        drift_warnings = TermDriftChecker().check_befund(
            befund_text=befund_text_for_drift,
            regulatorik=self.regulatorik,
            retrieved_chunks=retrieved_chunk_texts,
        )
        if drift_warnings:
            val_warnings.extend(drift_warnings)
            review_erforderlich = True

        # 9. Per-Claim Provenance Annotation
        begruendung_text = llm_result.get("begruendung", "") or ""
        claim_prov = annotate_claims(begruendung_text, good_nodes)

        befund = Befund(
            prueffeld_id=prueffeld["id"],
            frage=prueffeld["frage"],
            bewertung=Bewertung(bewertung_str),
            begruendung=begruendung_text,
            belegte_textstellen=llm_result.get("belegte_textstellen", []),
            empfehlungen=llm_result.get("empfehlungen", []),
            mangel_text=llm_result.get("mangel_text"),
            schweregrad=prueffeld.get("schweregrad"),
            quellen=llm_result.get("quellen", []),
            confidence=confidence,
            confidence_level=confidence_level,
            confidence_guards=guard_result,
            low_confidence_reasons=low_confidence_reasons,
            token_usage=token_usage,
            claim_list=build_claim_annotations(
                llm_result, good_nodes, retrieved_sources
            ),
            review_erforderlich=review_erforderlich,
            validierungshinweise=val_warnings,
            term_drift_warnings=drift_warnings,
            claim_provenance=claim_prov,
        )
        if adv_ergebnis is not None:
            befund = _merge_adversarial(befund, adv_ergebnis)
        return befund

    # ------------------------------------------------------------------ #
    # Retrieval
    # ------------------------------------------------------------------ #

    def _retrieve_evidence(self, prueffeld: dict) -> list:
        """Baut eine optimierte Suchanfrage und holt relevante Chunks."""
        parts = [prueffeld["frage"]]
        if prueffeld.get("erwartete_evidenz"):
            parts.append(
                f"Relevante Begriffe: {', '.join(prueffeld['erwartete_evidenz'])}"
            )
        rg = prueffeld.get("rechtsgrundlagen", [])
        if isinstance(rg, list) and rg:
            parts.append(f"Rechtsgrundlage: {', '.join(rg)}")
        elif isinstance(rg, str) and rg:
            parts.append(f"Rechtsgrundlage: {rg}")
        return self.retriever.retrieve("\n".join(parts))

    def _classify_evidence_chunk(self, text: str) -> tuple[str, str | None]:
        if REG_GUARDRAIL_RE.search(text or ""):
            return "regulatory_requirement", None
        lower = (text or "").lower()
        if any(term in lower for term in MARKETING_TERMS):
            return "context_noise", "MARKETING_PHRASE"
        if any(term in lower for term in CONTROL_TERMS):
            return "control_evidence", None
        if any(term in lower for term in NON_CONTROL_TERMS):
            return "context_noise", "NON_CONTROL_CONTEXT"
        return "context_noise", "NO_REG_REF"

    def _apply_relevance_filter(
        self, nodes: list, prueffeld: dict
    ) -> tuple[list, list]:
        kept = []
        dropped = []
        for node in nodes:
            text = node.get_content()
            klassifikation, reason = self._classify_evidence_chunk(text)
            if klassifikation == "context_noise" and reason:
                md = node.metadata or {}
                dropped.append(
                    {
                        "prueffeld_id": prueffeld.get("id"),
                        "source": md.get("source", "unbekannt"),
                        "reason": reason,
                        "snippet": (text or "")[:220],
                    }
                )
                continue
            kept.append(node)
        return kept, dropped

    def _format_evidence(self, nodes: list, prueffeld: dict) -> str:
        """Formatiert die Evidenz für den LLM-Prompt."""
        if not nodes:
            return (
                "KEINE EVIDENZ GEFUNDEN – keine relevanten Dokumente im Prüfungskorpus."
            )

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
                lines.append(
                    f"[Screenshot-Datei: {source} – visuelle Prüfung durch Mensch erforderlich]"
                )
            else:
                lines.append(node.get_content()[:2000])
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # LLM-Bewertung
    # ------------------------------------------------------------------ #

    def _adversarial_evaluate(
        self, prueffeld: dict, evidenz_text: str
    ) -> AdversarialErgebnis:
        """Zweiter Pass mit kritisch-adversarialem Prompt auf identischer Evidenz."""
        rg = prueffeld.get("rechtsgrundlagen", [])
        if isinstance(rg, list):
            rg_str = ", ".join(rg)
        else:
            rg_str = str(rg)

        user_prompt = f"""## PRÜFFELD (adversarial): {prueffeld["id"]}
**Frage:** {prueffeld["frage"]}
**Rechtsgrundlage:** {rg_str}
**Erwartete Evidenz:** {", ".join(prueffeld.get("erwartete_evidenz", []))}
**Schweregrad:** {prueffeld.get("schweregrad", "unbekannt")}
**Bewertungskriterien:** {prueffeld.get("bewertungskriterien", "")}

{evidenz_text}

Finde alle Schwachstellen. Antworte als JSON.
"""
        messages = [
            SystemMessage(content=self.adversarial_system_prompt),
            HumanMessage(content=user_prompt),
        ]

        result = {
            "bewertung": "nicht_prüfbar",
            "schwachstellen": [],
            "fehlende_nachweise": [],
        }
        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = self.llm.invoke(messages)
                result = extract_json(response.content)
                break
            except json.JSONDecodeError:
                logger.warning(
                    "Adversarial-Antwort für %s nicht parsebar; fallback auf nicht_prüfbar",
                    prueffeld["id"],
                )
                break
            except Exception as e:
                if attempt < LLM_MAX_RETRIES - 1:
                    time.sleep(LLM_RETRY_BASE_DELAY * (2**attempt))
                else:
                    logger.warning(
                        "Adversarial-Pass für %s fehlgeschlagen: %s",
                        prueffeld["id"],
                        e,
                    )

        try:
            adv_bewertung = Bewertung(result.get("bewertung", "nicht_prüfbar"))
        except ValueError:
            adv_bewertung = Bewertung.NICHT_PRUEFBAR

        return AdversarialErgebnis(
            prueffeld_id=prueffeld["id"],
            adversarial_bewertung=adv_bewertung,
            schwachstellen=result.get("schwachstellen", []),
            fehlende_nachweise=result.get("fehlende_nachweise", []),
        )

    def _evaluate_with_llm(self, prueffeld: dict, evidenz_text: str) -> dict:
        """Sendet Prüffrage + Evidenz an das LLM und parst das JSON-Ergebnis."""
        rg = prueffeld.get("rechtsgrundlagen", [])
        if isinstance(rg, list):
            rg_str = ", ".join(rg)
        else:
            rg_str = str(rg)

        user_prompt = f"""## PRÜFFELD: {prueffeld["id"]}
**Frage:** {prueffeld["frage"]}
**Rechtsgrundlage:** {rg_str}
**Erwartete Evidenz:** {", ".join(prueffeld.get("erwartete_evidenz", []))}
**Schweregrad:** {prueffeld.get("schweregrad", "unbekannt")}
**Bewertungskriterien:** {prueffeld.get("bewertungskriterien", "")}

{evidenz_text}

Bewerte dieses Prüffeld und antworte als JSON.
"""
        prompt_input_tokens = estimate_tokens(self.system_prompt) + estimate_tokens(
            user_prompt
        )
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=user_prompt),
        ]

        last_exc: Optional[Exception] = None
        for attempt in range(LLM_MAX_RETRIES):
            try:
                response = self.llm.invoke(messages)
                parsed = extract_json(response.content)
                output_tokens = estimate_tokens(response.content)
                parsed["_token_usage"] = {
                    "input": prompt_input_tokens,
                    "output": output_tokens,
                    "total": prompt_input_tokens + output_tokens,
                }
                return parsed
            except json.JSONDecodeError as e:
                # Parse-Fehler: Retry hilft hier nicht
                return {
                    "bewertung": "nicht_prüfbar",
                    "begruendung": f"LLM-Antwort konnte nicht als JSON geparst werden: {e}",
                    "belegte_textstellen": [],
                    "mangel_text": None,
                    "empfehlungen": [],
                    "quellen": [],
                    "confidence_self": 0.0,
                    "_token_usage": {
                        "input": prompt_input_tokens,
                        "output": 0,
                        "total": prompt_input_tokens,
                    },
                }
            except Exception as e:
                last_exc = e
                if attempt < LLM_MAX_RETRIES - 1:
                    delay = LLM_RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "LLM-Aufruf für %s fehlgeschlagen (%s: %s) – Retry %d/%d in %.0fs",
                        prueffeld["id"],
                        type(e).__name__,
                        e,
                        attempt + 1,
                        LLM_MAX_RETRIES - 1,
                        delay,
                    )
                    time.sleep(delay)

        return {
            "bewertung": "nicht_prüfbar",
            "begruendung": (
                f"API-Fehler nach {LLM_MAX_RETRIES} Versuchen: "
                f"{type(last_exc).__name__}: {last_exc}"
            ),
            "belegte_textstellen": [],
            "mangel_text": None,
            "empfehlungen": [],
            "quellen": [],
            "confidence_self": 0.0,
            "_token_usage": {
                "input": prompt_input_tokens,
                "output": 0,
                "total": prompt_input_tokens,
            },
        }


# ------------------------------------------------------------------ #
# Adversarial Merge
# ------------------------------------------------------------------ #


def _merge_adversarial(befund: Befund, adv: AdversarialErgebnis) -> Befund:
    """
    Führt originalen Befund und adversariales Ergebnis zusammen.

    Divergenz-Logik:
      0   → Adversarial bestätigt → kein Eingriff
      1   → Leicht strenger → kleine Confidence-Penalty + Hinweis
      ≥2  → Wesentlich strenger → größere Penalty + Review erzwingen
    """
    sev_normal = BEWERTUNG_SEVERITY.get(befund.bewertung.value, 3)
    sev_adv = BEWERTUNG_SEVERITY.get(adv.adversarial_bewertung.value, 3)
    divergenz = sev_adv - sev_normal  # positiv = adversarial strenger

    hinweise = list(befund.validierungshinweise)
    confidence_delta = 0.0
    review_erforderlich = befund.review_erforderlich
    neue_bewertung = befund.bewertung
    disputed_positions: Optional[dict] = None

    if divergenz <= 0:
        hinweise.append(
            f"⚔️ Adversarial Layer bestätigt Bewertung "
            f"({adv.adversarial_bewertung.value})"
        )
    elif divergenz == 1:
        confidence_delta = _ADVERSARIAL_PENALTY[1]
        hinweise.append(
            f"⚔️ Adversarial Layer schätzt strenger: "
            f"{befund.bewertung.value} → {adv.adversarial_bewertung.value} "
            f"(Divergenz: {divergenz})"
        )
        for sw in adv.schwachstellen:
            hinweise.append(f"⚔️ Schwachstelle: {sw}")
    else:
        confidence_delta = _ADVERSARIAL_PENALTY.get(divergenz, _ADVERSARIAL_PENALTY[3])
        review_erforderlich = True

        # Disputed-Status: divergenz >= 2 → wesentliche Uneinigkeit
        if divergenz >= 2:
            neue_bewertung = Bewertung.DISPUTED
            review_erforderlich = True
            disputed_positions = {
                "pruefer": befund.bewertung.value,
                "adversarial": adv.adversarial_bewertung.value,
                "divergenz": divergenz,
            }
            hinweise.append(
                f"⚔️ DISPUTED: Adversarial Divergenz ({divergenz}): "
                f"{befund.bewertung.value} → {adv.adversarial_bewertung.value} "
                f"→ Bewertung auf 'disputed' gesetzt, Review erzwungen"
            )
        else:
            hinweise.append(
                f"⚔️ Adversarial Divergenz ({divergenz}): "
                f"{befund.bewertung.value} → {adv.adversarial_bewertung.value} "
                f"→ Review erzwungen"
            )
        for sw in adv.schwachstellen:
            hinweise.append(f"⚔️ Schwachstelle: {sw}")
        for fn in adv.fehlende_nachweise:
            hinweise.append(f"📄 Fehlender Nachweis (adversarial): {fn}")

    new_confidence = max(0.0, round(befund.confidence - confidence_delta, 3))

    return Befund(
        prueffeld_id=befund.prueffeld_id,
        frage=befund.frage,
        bewertung=neue_bewertung,
        begruendung=befund.begruendung,
        belegte_textstellen=befund.belegte_textstellen,
        empfehlungen=befund.empfehlungen,
        mangel_text=befund.mangel_text,
        schweregrad=befund.schweregrad,
        quellen=befund.quellen,
        confidence=new_confidence,
        confidence_level=befund.confidence_level,
        confidence_guards=befund.confidence_guards,
        low_confidence_reasons=befund.low_confidence_reasons,
        token_usage=befund.token_usage,
        claim_list=befund.claim_list,
        review_erforderlich=review_erforderlich,
        validierungshinweise=hinweise,
        term_drift_warnings=befund.term_drift_warnings,
        claim_provenance=befund.claim_provenance,
        disputed_positions=disputed_positions,
    )


# Rückwärtskompatibilität
GwGPrueferAgent = PrueferAgent
Sektionsergebniss = Sektionsergebnis  # früherer Typo-Alias
