"""
FinRegAgents – Skeptiker-Agent (v1)
Adversariales Post-Processing: Hinterfragt die Befunde des PrueferAgent.

Der Skeptiker-Agent ist ein zweiter LLM-Aufruf, der:
1. Den Befund des Prüfer-Agents aktiv herausfordert
2. Besonders bei 'konform'-Ratings schwache Evidenz aufdeckt
3. Bei überkonfidenten Bewertungen eskaliert
4. Eine eigene Bewertungsempfehlung und Einwände liefert

Architektur:
    PrueferAgent → Befund → SkeptikerAgent → SkeptikerBefund

Integration:
    Optionaler Post-Processing-Layer in der Pipeline.
    Wird nur aufgerufen wenn Confidence > SKEPTIKER_MIN_CONFIDENCE
    (niedrig-confidence Befunde sind bereits als 'Review erforderlich' markiert).
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents.pruef_agent import (
    Befund,
    Bewertung,
    confidence_level_from_score,
    estimate_tokens,
    extract_json,
)

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Konfiguration
# ------------------------------------------------------------------ #

# Ab diesem Confidence-Wert wird der Skeptiker aufgerufen
# (unter diesem Wert ist der Befund bereits als 'Review erforderlich' markiert)
SKEPTIKER_MIN_CONFIDENCE = 0.5

# Confidence-Penalty wenn Skeptiker Einwände hat
SKEPTIKER_CONFIDENCE_PENALTY = 0.15

# Schwellenwert: Ab wievielen Einwänden wird eine Bewertungs-Eskalation empfohlen
EINWAND_ESKALATION_THRESHOLD = 2

# Retry-Konfiguration (analog PrueferAgent)
SKEPTIKER_MAX_RETRIES = 3
SKEPTIKER_RETRY_BASE_DELAY = 2.0


def _to_bool(value, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "ja"}:
            return True
        if normalized in {"false", "0", "no", "nein"}:
            return False
    return default


def _to_str_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple, set)):
        result = []
        for item in value:
            if item is None:
                continue
            text = str(item).strip()
            if text:
                result.append(text)
        return result
    text = str(value).strip()
    return [text] if text else []


# ------------------------------------------------------------------ #
# Datenmodelle
# ------------------------------------------------------------------ #


@dataclass
class SkeptikerBefund:
    """Ergebnis des Skeptiker-Reviews."""

    prueffeld_id: str
    original_bewertung: Bewertung
    original_confidence: float

    # Skeptiker-Einschätzung
    akzeptiert: bool  # Hat Skeptiker die orig. Bewertung akzeptiert?
    bewertung_empfehlung: Optional[
        Bewertung
    ]  # Empfehlung des Skeptikers (None = akzeptiert)
    einwaende: list[str] = field(default_factory=list)
    staerken: list[str] = field(default_factory=list)
    schweregrad_erhoehen: bool = False  # Sollte Schweregrad erhöht werden?
    nachforderung_empfohlen: bool = False  # Fehlende Dokumente nachfordern?
    fehlende_evidenz: list[str] = field(default_factory=list)

    # Angepasster Confidence-Score
    adjustierter_confidence: float = 0.0
    token_usage: dict = field(default_factory=dict)

    # Rohe LLM-Antwort
    skeptiker_raw: dict = field(default_factory=dict)


# ------------------------------------------------------------------ #
# System-Prompt
# ------------------------------------------------------------------ #

SKEPTIKER_SYSTEM_PROMPT = """Du bist ein erfahrener Revisionsleiter, der die Arbeit von Prüfern kritisch hinterfragt.
Deine Aufgabe: Finde Schwächen und blinde Flecken in einer bereits erstellten Prüfungsbewertung.

Du bist KEIN Prüfer – du bist der Advocatus Diaboli. Dein Job ist es, die Bewertung
des Prüfers so streng wie möglich zu challengen. Dabei gilt:

**Bei 'konform'-Ratings hinterfragst du:**
- Ist die Evidenz wirklich belastbar, oder nur eine formale Hülle ohne Substanz?
- Gibt es Hinweise auf "Papierkonformität" ohne echte Umsetzung?
- Sind die zitierten Textstellen ausreichend spezifisch oder zu vage?
- Fehlen kritische Dokumenttypen (z.B. Audit-Trail, Schulungsnachweise)?
- Widerspricht die Bewertung dem Schweregrad des Prüffelds?

**Bei 'nicht_konform'/'teilkonform'-Ratings prüfst du:**
- Wurden mildernde Faktoren angemessen berücksichtigt?
- Ist die Schwere der Mängel verhältnismäßig bewertet?
- Gibt es Hinweise auf kompensatorische Kontrollen?

**Grundsätze:**
- Sei konstruktiv kritisch, nicht destruktiv
- Benenne konkret WELCHE Evidenz fehlt oder schwach ist
- Unterscheide zwischen "schwacher Evidenz" und "fehlender Evidenz"
- Regulatorische Mindeststandards haben Vorrang vor hausinternen Regeln
- Formale Dokumente ohne Prozessnachweis gelten als unzureichend

Antworte AUSSCHLIESSLICH als valides JSON mit dieser Struktur:
{
  "akzeptiert": true/false,
  "bewertung_empfehlung": "konform|teilkonform|nicht_konform|nicht_prüfbar|null",
  "einwaende": ["Einwand 1", "Einwand 2"],
  "staerken": ["Stärke 1 der Bewertung"],
  "schweregrad_erhoehen": true/false,
  "nachforderung_empfohlen": true/false,
  "fehlende_evidenz": ["Typ 1 fehlt", "Dokument X fehlt"],
  "begruendung": "Zusammenfassung der Skeptiker-Einschätzung (3-5 Sätze)"
}

Wenn du akzeptierst (akzeptiert=true), setze bewertung_empfehlung auf null.
"""


# ------------------------------------------------------------------ #
# Skeptiker-Agent
# ------------------------------------------------------------------ #


class SkeptikerAgent:
    """
    Adversarialer Post-Processing-Agent. Hinterfragt Befunde des PrueferAgent.

    Wird nur für Befunde aufgerufen die:
    1. Ausreichend Confidence haben (über SKEPTIKER_MIN_CONFIDENCE)
    2. Nicht bereits als 'nicht_prüfbar' bewertet wurden
    3. Relevant für adversariales Review sind (konform + hohe Confidence)
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-5-20250514",
        temperature: float = 0.3,  # Etwas höher als PrueferAgent für kreativere Einwände
        only_konform: bool = False,  # Nur konform-Ratings challengen
    ):
        self.llm = ChatAnthropic(model=model, temperature=temperature, max_tokens=1500)
        self.only_konform = only_konform

    def reviewe(
        self,
        befund: Befund,
        prueffeld: dict,
        evidenz_text: str = "",
    ) -> SkeptikerBefund:
        """
        Hauptmethode: Führt adversariales Review eines Befunds durch.

        Args:
            befund: Originaler Befund des PrueferAgent
            prueffeld: Das Prüffeld-Dict aus dem Katalog
            evidenz_text: Die Evidenz, die dem PrueferAgent vorlag (optional)

        Returns:
            SkeptikerBefund mit Einwänden und ggf. geänderter Empfehlung
        """
        # Nicht für nicht_prüfbar aufrufen
        if befund.bewertung == Bewertung.NICHT_PRUEFBAR:
            return self._pass_through(befund)

        # Bei only_konform: nur konform challengen
        if self.only_konform and befund.bewertung != Bewertung.KONFORM:
            return self._pass_through(befund)

        # Confidence zu niedrig → bereits als review markiert, kein Skeptiker nötig
        if befund.confidence < SKEPTIKER_MIN_CONFIDENCE:
            return self._pass_through(befund)

        # Adversariales Review durchführen
        skeptiker_result = self._challenge(befund, prueffeld, evidenz_text)
        return self._build_skeptiker_befund(befund, skeptiker_result)

    def _challenge(
        self,
        befund: Befund,
        prueffeld: dict,
        evidenz_text: str,
    ) -> dict:
        """Sendet Befund + Kontext an den Skeptiker-LLM."""
        # Befund für den Skeptiker aufbereiten
        befund_summary = f"""## ORIGINAL-BEWERTUNG
**Prüffeld:** {befund.prueffeld_id} – {befund.frage}
**Bewertung:** {befund.bewertung.value.upper()}
**Confidence:** {befund.confidence:.0%}
**Schweregrad:** {befund.schweregrad or "nicht angegeben"}

**Begründung des Prüfers:**
{befund.begruendung}

**Belegte Textstellen:**
{chr(10).join(f"- {t}" for t in befund.belegte_textstellen) if befund.belegte_textstellen else "(keine)"}

**Mangel-Text:**
{befund.mangel_text or "(kein Mangel formuliert)"}

**Quellen:**
{", ".join(befund.quellen) if befund.quellen else "(keine angegeben)"}

**Validierungshinweise (automatisch):**
{chr(10).join(f"- {h}" for h in befund.validierungshinweise) if befund.validierungshinweise else "(keine)"}
"""

        katalog_kontext = f"""## PRÜFFELD-KONTEXT (aus Katalog)
**Erwartete Evidenz:** {", ".join(prueffeld.get("erwartete_evidenz", []))}
**Erlaubte Dokumenttypen:** {", ".join(prueffeld.get("input_typen", []))}
**Bewertungskriterien:** {prueffeld.get("bewertungskriterien", "nicht angegeben")}
**Rechtsgrundlagen:** {", ".join(prueffeld.get("rechtsgrundlagen", []))}
"""

        evidenz_kontext = ""
        if evidenz_text:
            # Evidenz kürzen um Token zu sparen (Skeptiker braucht nur Zusammenfassung)
            evidenz_kontext = f"\n## VERFÜGBARE EVIDENZ (Auszug)\n{evidenz_text[:3000]}"

        user_prompt = f"""{befund_summary}

{katalog_kontext}
{evidenz_kontext}

**Deine Aufgabe:** Challenge diese Bewertung. Finde Schwächen.
Bei einer '{befund.bewertung.value}'-Bewertung – bist du damit einverstanden?
Antworte als JSON.
"""
        prompt_input_tokens = estimate_tokens(SKEPTIKER_SYSTEM_PROMPT) + estimate_tokens(user_prompt)
        messages = [
            SystemMessage(content=SKEPTIKER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        import json as _json

        last_exc: Optional[Exception] = None
        for attempt in range(SKEPTIKER_MAX_RETRIES):
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
            except _json.JSONDecodeError as e:
                # Parse-Fehler → kein Retry sinnvoll
                logger.warning(
                    "Skeptiker JSON-Parse-Fehler für %s: %s", befund.prueffeld_id, e
                )
                return {
                    "akzeptiert": True,
                    "bewertung_empfehlung": None,
                    "einwaende": [f"Skeptiker-Review: JSON-Parse-Fehler ({e})"],
                    "staerken": [],
                    "schweregrad_erhoehen": False,
                    "nachforderung_empfohlen": False,
                    "fehlende_evidenz": [],
                    "begruendung": "Skeptiker-Antwort konnte nicht geparst werden.",
                    "_token_usage": {"input": prompt_input_tokens, "output": 0, "total": prompt_input_tokens},
                }
            except Exception as e:
                last_exc = e
                if attempt < SKEPTIKER_MAX_RETRIES - 1:
                    delay = SKEPTIKER_RETRY_BASE_DELAY * (2**attempt)
                    logger.warning(
                        "Skeptiker-LLM-Aufruf für %s fehlgeschlagen (Versuch %d/%d), "
                        "Retry in %.0fs: %s",
                        befund.prueffeld_id,
                        attempt + 1,
                        SKEPTIKER_MAX_RETRIES,
                        delay,
                        e,
                    )
                    time.sleep(delay)

        logger.error(
            "Skeptiker-LLM für %s nach %d Versuchen fehlgeschlagen: %s",
            befund.prueffeld_id,
            SKEPTIKER_MAX_RETRIES,
            last_exc,
        )
        return {
            "akzeptiert": True,
            "bewertung_empfehlung": None,
            "einwaende": [
                f"Skeptiker-Review fehlgeschlagen: {type(last_exc).__name__}: {last_exc}"
            ],
            "staerken": [],
            "schweregrad_erhoehen": False,
            "nachforderung_empfohlen": False,
            "fehlende_evidenz": [],
            "begruendung": "Skeptiker-Review konnte nicht durchgeführt werden.",
            "_token_usage": {"input": prompt_input_tokens, "output": 0, "total": prompt_input_tokens},
        }

    def _build_skeptiker_befund(self, befund: Befund, result: dict) -> SkeptikerBefund:
        """Baut den SkeptikerBefund aus dem LLM-Ergebnis."""
        token_usage = result.pop("_token_usage", {"input": 0, "output": 0, "total": 0})
        akzeptiert = _to_bool(result.get("akzeptiert", True), True)
        einwaende = _to_str_list(result.get("einwaende", []))
        staerken = _to_str_list(result.get("staerken", []))
        fehlende_evidenz = _to_str_list(result.get("fehlende_evidenz", []))
        schweregrad_erhoehen = _to_bool(
            result.get("schweregrad_erhoehen", False), False
        )
        nachforderung_empfohlen = _to_bool(
            result.get("nachforderung_empfohlen", False), False
        )

        # Bewertungsempfehlung nur wenn nicht akzeptiert
        empfehlung = None
        if not akzeptiert:
            empf_str = result.get("bewertung_empfehlung")
            if empf_str and empf_str != "null":
                try:
                    empfehlung = Bewertung(str(empf_str).strip().lower())
                except ValueError:
                    pass

        # Confidence-Penalty bei Einwänden
        penalty = SKEPTIKER_CONFIDENCE_PENALTY * len(einwaende) if einwaende else 0.0
        adjustierter_confidence = max(0.0, round(befund.confidence - penalty, 3))

        return SkeptikerBefund(
            prueffeld_id=befund.prueffeld_id,
            original_bewertung=befund.bewertung,
            original_confidence=befund.confidence,
            akzeptiert=akzeptiert,
            bewertung_empfehlung=empfehlung,
            einwaende=einwaende,
            staerken=staerken,
            schweregrad_erhoehen=schweregrad_erhoehen,
            nachforderung_empfohlen=nachforderung_empfohlen,
            fehlende_evidenz=fehlende_evidenz,
            adjustierter_confidence=adjustierter_confidence,
            token_usage=token_usage,
            skeptiker_raw=result,
        )

    def _pass_through(self, befund: Befund) -> SkeptikerBefund:
        """Erstellt einen SkeptikerBefund ohne Review (Pass-through)."""
        return SkeptikerBefund(
            prueffeld_id=befund.prueffeld_id,
            original_bewertung=befund.bewertung,
            original_confidence=befund.confidence,
            akzeptiert=True,
            bewertung_empfehlung=None,
            adjustierter_confidence=befund.confidence,
            token_usage={"input": 0, "output": 0, "total": 0},
        )

    def reviewe_sektionsergebnis(
        self,
        sektionsergebnis,
        katalog_sektionen: dict,
        evidenz_map: dict = None,
    ) -> dict:
        """
        Führt Skeptiker-Review für alle Befunde einer Sektion durch.

        Args:
            sektionsergebnis: Sektionsergebnis-Objekt
            katalog_sektionen: Dict {prueffeld_id: prueffeld_dict}
            evidenz_map: Optional Dict {prueffeld_id: evidenz_text}

        Returns:
            Dict {prueffeld_id: SkeptikerBefund}
        """
        ergebnisse = {}
        for befund in sektionsergebnis.befunde:
            prueffeld = katalog_sektionen.get(
                befund.prueffeld_id,
                {
                    "id": befund.prueffeld_id,
                    "frage": befund.frage,
                    "erwartete_evidenz": [],
                    "input_typen": [],
                    "bewertungskriterien": "",
                    "rechtsgrundlagen": [],
                },
            )
            evidenz = (evidenz_map or {}).get(befund.prueffeld_id, "")
            skeptiker_befund = self.reviewe(befund, prueffeld, evidenz)
            ergebnisse[befund.prueffeld_id] = skeptiker_befund

        return ergebnisse


# ------------------------------------------------------------------ #
# Hilfsfunktion: Befund mit Skeptiker-Ergebnis zusammenführen
# ------------------------------------------------------------------ #


def merge_befund_skeptiker(befund: Befund, skeptiker: SkeptikerBefund) -> Befund:
    """
    Führt originalen Befund und Skeptiker-Review zusammen.
    Aktualisiert Confidence, Validierungshinweise und ggf. Review-Status.
    Der Befund behält seine originale Bewertung – der Skeptiker gibt nur eine Empfehlung.
    """
    hinweise = list(befund.validierungshinweise)

    if not skeptiker.akzeptiert:
        hinweise.append(
            f"⚔️ Skeptiker widerspricht: Empfehlung '{skeptiker.bewertung_empfehlung.value if skeptiker.bewertung_empfehlung else 'unklar'}'"
        )
        for einwand in skeptiker.einwaende:
            hinweise.append(f"⚔️ Einwand: {einwand}")
        for fehlend in skeptiker.fehlende_evidenz:
            hinweise.append(f"📄 Fehlende Evidenz: {fehlend}")
    elif skeptiker.einwaende:
        for einwand in skeptiker.einwaende:
            hinweise.append(f"⚠️ Skeptiker-Hinweis: {einwand}")

    if skeptiker.schweregrad_erhoehen:
        hinweise.append("⬆️ Skeptiker empfiehlt: Schweregrad erhöhen")

    if skeptiker.nachforderung_empfohlen:
        hinweise.append("📋 Skeptiker empfiehlt: Dokumente nachfordern")

    # Review erzwingen wenn Skeptiker nicht akzeptiert oder Einwände hat
    review_erforderlich = (
        befund.review_erforderlich
        or not skeptiker.akzeptiert
        or len(skeptiker.einwaende) >= EINWAND_ESKALATION_THRESHOLD
    )

    return Befund(
        prueffeld_id=befund.prueffeld_id,
        frage=befund.frage,
        bewertung=befund.bewertung,  # Originalbewertung behalten!
        begruendung=befund.begruendung,
        belegte_textstellen=befund.belegte_textstellen,
        empfehlungen=befund.empfehlungen,
        mangel_text=befund.mangel_text,
        schweregrad=befund.schweregrad,
        quellen=befund.quellen,
        confidence=skeptiker.adjustierter_confidence,  # Angepasste Confidence
        confidence_level=confidence_level_from_score(skeptiker.adjustierter_confidence),
        confidence_guards=befund.confidence_guards,
        low_confidence_reasons=befund.low_confidence_reasons,
        token_usage=befund.token_usage,
        review_erforderlich=review_erforderlich,
        validierungshinweise=hinweise,
    )
