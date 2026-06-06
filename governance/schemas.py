"""
Block B – Schema-as-Contract.

Pydantic-Verträge für Agenten-Outputs: Pflichtfelder, Wertebereiche, `extra="forbid"`,
und maschinell durchgesetzte Business-Regeln (z. B. confidence < Schwelle ⇒ review).

Das ersetzt nicht die bestehenden @dataclass-Strukturen, sondern legt ein
*Validierungs-Gate* darüber: Befunde, die den Vertrag verletzen, werden blockiert/markiert,
statt unbemerkt weiterverarbeitet zu werden (Buch Kap. 2: „Freier Text ist kein
Bankschnittstellenformat").
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Schwelle gespiegelt aus agents.pruef_agent.CONFIDENCE_REVIEW_THRESHOLD
CONFIDENCE_REVIEW_THRESHOLD = 0.7

_BEWERTUNGEN = {"konform", "teilkonform", "nicht_konform", "nicht_prüfbar", "disputed"}
_SCHWEREGRADE = {"wesentlich", "bedeutsam", "gering", None}


class BefundContract(BaseModel):
    """Verbindlicher Output-Vertrag für einen einzelnen Befund."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    prueffeld_id: str = Field(min_length=1)
    frage: str = Field(min_length=1)
    bewertung: str
    begruendung: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_level: str = "low"
    review_erforderlich: bool = False
    belegte_textstellen: list[str] = Field(default_factory=list)
    quellen: list[str] = Field(default_factory=list)
    schweregrad: Optional[str] = None
    mangel_text: Optional[str] = None

    @field_validator("bewertung")
    @classmethod
    def _valid_bewertung(cls, v: str) -> str:
        if v not in _BEWERTUNGEN:
            raise ValueError(
                f"unzulässige Bewertung: {v!r}; erlaubt: {sorted(_BEWERTUNGEN)}"
            )
        return v

    @field_validator("schweregrad")
    @classmethod
    def _valid_schweregrad(cls, v):
        if v not in _SCHWEREGRADE:
            raise ValueError(f"unzulässiger Schweregrad: {v!r}")
        return v

    @model_validator(mode="after")
    def _enforce_review_rule(self):
        """Business-Regel: niedrige Confidence ⇒ Review-Pflicht (nicht optional)."""
        if (
            self.confidence < CONFIDENCE_REVIEW_THRESHOLD
            and not self.review_erforderlich
        ):
            raise ValueError(
                f"confidence {self.confidence:.2f} < {CONFIDENCE_REVIEW_THRESHOLD} "
                "verlangt review_erforderlich=True"
            )
        # Mängel müssen bei Nicht-Konformität benannt sein
        if self.bewertung in ("nicht_konform", "teilkonform") and not (
            self.mangel_text or self.begruendung
        ):
            raise ValueError(
                "nicht_konform/teilkonform erfordert mangel_text oder begruendung"
            )
        return self


class ValidationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prueffeld_id: str
    ok: bool
    errors: list[str] = Field(default_factory=list)


def _befund_to_dict(befund) -> dict:
    """Duck-Typing-Bridge: dataclass Befund → dict für den Vertrag."""
    bew = getattr(befund, "bewertung", None)
    bew = getattr(bew, "value", bew)
    return {
        "prueffeld_id": getattr(befund, "prueffeld_id", ""),
        "frage": getattr(befund, "frage", ""),
        "bewertung": bew,
        "begruendung": getattr(befund, "begruendung", ""),
        "confidence": float(getattr(befund, "confidence", 0.0) or 0.0),
        "confidence_level": getattr(befund, "confidence_level", "low"),
        "review_erforderlich": bool(getattr(befund, "review_erforderlich", False)),
        "belegte_textstellen": list(getattr(befund, "belegte_textstellen", []) or []),
        "quellen": list(getattr(befund, "quellen", []) or []),
        "schweregrad": getattr(befund, "schweregrad", None),
        "mangel_text": getattr(befund, "mangel_text", None),
    }


def validate_befund(befund) -> ValidationResult:
    """Validiert einen Befund (dataclass ODER dict) gegen den Vertrag. Wirft nie."""
    data = befund if isinstance(befund, dict) else _befund_to_dict(befund)
    pf = data.get("prueffeld_id", "?")
    try:
        BefundContract(**data)
        return ValidationResult(prueffeld_id=pf, ok=True)
    except Exception as e:  # pydantic.ValidationError o. ä.
        msgs = [
            str(err.get("msg", err)) for err in getattr(e, "errors", lambda: [])()
        ] or [str(e)]
        return ValidationResult(prueffeld_id=pf, ok=False, errors=msgs)


def validate_run(sektionsergebnisse) -> dict:
    """Validiert alle Befunde eines Laufs. Gibt Aggregat + Verstöße zurück."""
    results: list[ValidationResult] = []
    for sektion in sektionsergebnisse or []:
        for befund in getattr(sektion, "befunde", []) or []:
            results.append(validate_befund(befund))
    total = len(results)
    ok = sum(1 for r in results if r.ok)
    return {
        "total": total,
        "valid": ok,
        "invalid": total - ok,
        "schema_compliance": round(ok / total, 4) if total else 1.0,
        "violations": [r.model_dump() for r in results if not r.ok],
    }
