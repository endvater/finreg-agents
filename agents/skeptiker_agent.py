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

from dataclasses import dataclass, field
from typing import Optional
import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents.pruef_agent import (
    Befund,
    Bewertung,
    extract_json,
    CONFIDENCE_REVIEW_THRESHOLD,
)


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
    akzeptiert: bool                          # Hat Skeptiker die orig. Bewertung akzeptiert?
    bewertung_empfehlung: Optional[Bewertung]  # Empfehlung des Skeptikers (None = akzeptiert)
    einwaende: list[str] = field(default_factory=list)
    staerken: list[str] = field(default_factory=list)
    schweregrad_erhoehen: bool = False        # Sollte Schweregrad erhöht werden?
    nachforderung_empfohlen: bool = False     # Fehlende Dokumente nachfordern?
    fehlende_evidenz: list[str] = field(default_factory=list)

    # Angepasster Confidence-Score
    adjustierter_confidence: float = 0.0

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
        temperature: float = 0.3,   # Etwas höher als PrueferAgent für kreativere Einwände
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
**Schweregrad:** {befund.schweregrad or 'nicht angegeben'}

**Begründung des Prüfers:**
{befund.begruendung}

**Belegte Textstellen:**
{chr(10).join(f'- {t}' for t in befund.belegte_textstellen) if befund.belegte_textstellen else '(keine)'}

**Mangel-Text:**
{befund.mangel_text or '(kein Mangel formuliert)'}

**Quellen:**
{', '.join(befund.quellen) if befund.quellen else '(keine angegeben)'}

**Validierungshinweise (automatisch):**
{chr(10).join(f'- {h}' for h in befund.validierungshinweise) if befund.validierungshinweise else '(keine)'}
"""

        katalog_kontext = f"""## PRÜFFELD-KONTEXT (aus Katalog)
**Erwartete Evidenz:** {', '.join(prueffeld.get('erwartete_evidenz', []))}
**Erlaubte Dokumenttypen:** {', '.join(prueffeld.get('input_typen', []))}
**Bewertungskriterien:** {prueffeld.get('bewertungskriterien', 'nicht angegeben')}
**Rechtsgrundlagen:** {', '.join(prueffeld.get('rechtsgrundlagen', []))}
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
        messages = [
            SystemMessage(content=SKEPTIKER_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return extract_json(response.content)
        except Exception as e:
            return {
                "akzeptiert": True,
                "bewertung_empfehlung": None,
                "einwaende": [f"Skeptiker-Review fehlgeschlagen: {type(e).__name__}: {e}"],
                "staerken": [],
                "schweregrad_erhoehen": False,
                "nachforderung_empfohlen": False,
                "fehlende_evidenz": [],
                "begruendung": "Skeptiker-Review konnte nicht durchgeführt werden.",
            }

    def _build_skeptiker_befund(self, befund: Befund, result: dict) -> SkeptikerBefund:
        """Baut den SkeptikerBefund aus dem LLM-Ergebnis."""
        akzeptiert = result.get("akzeptiert", True)
        einwaende = result.get("einwaende", [])

        # Bewertungsempfehlung nur wenn nicht akzeptiert
        empfehlung = None
        if not akzeptiert:
            empf_str = result.get("bewertung_empfehlung")
            if empf_str and empf_str != "null":
                try:
                    empfehlung = Bewertung(empf_str)
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
            staerken=result.get("staerken", []),
            schweregrad_erhoehen=result.get("schweregrad_erhoehen", False),
            nachforderung_empfohlen=result.get("nachforderung_empfohlen", False),
            fehlende_evidenz=result.get("fehlende_evidenz", []),
            adjustierter_confidence=adjustierter_confidence,
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
            prueffeld = katalog_sektionen.get(befund.prueffeld_id, {
                "id": befund.prueffeld_id,
                "frage": befund.frage,
                "erwartete_evidenz": [],
                "input_typen": [],
                "bewertungskriterien": "",
                "rechtsgrundlagen": [],
            })
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
        bewertung=befund.bewertung,             # Originalbewertung behalten!
        begruendung=befund.begruendung,
        belegte_textstellen=befund.belegte_textstellen,
        empfehlungen=befund.empfehlungen,
        mangel_text=befund.mangel_text,
        schweregrad=befund.schweregrad,
        quellen=befund.quellen,
        confidence=skeptiker.adjustierter_confidence,  # Angepasste Confidence
        review_erforderlich=review_erforderlich,
        validierungshinweise=hinweise,
    )
