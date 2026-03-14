"""
FinRegAgents – Term Drift Checker (Issue #3)
Erkennt Terminologie-Drift in LLM-generierten Befunden gegenüber regulatorischen Quelltexten.

Phantom-Zitat: Eine Rechtsnorm (§ X, Art. Y, MaRisk XY) erscheint im Befundtext,
aber in keinem der für dieses Prüffeld abgerufenen Retrieval-Chunks.
"""

import re

# ------------------------------------------------------------------ #
# Regulatorik-spezifische Terminologie-Muster
# ------------------------------------------------------------------ #

TERM_PATTERNS: dict[str, list[str]] = {
    "gwg": [
        r"§\s*\d+[a-z]?\s+GwG",
        r"\bGwG\b",
        r"\bAML\b",
        r"\bKYC\b",
        r"§25[a-z]\s+KWG",
        r"Geldwäsche",
        r"Sorgfaltspflicht",
        r"Risikoanalyse",
        r"Identifizierungspflicht",
        r"Verdachtsmeldung",
    ],
    "dora": [
        r"Art\.\s*\d+\s+DORA",
        r"\bDORA\b",
        r"\bIKT\b",
        r"\bICT\b",
        r"digitale(?:r|s|n)?\s+Resilienz",
        r"Vorfallsmanagement",
        r"Drittanbieter",
        r"TIBER",
        r"RTS\s+ICT",
    ],
    "marisk": [
        r"MaRisk\s+[A-Z]{2}\s+\d",
        r"§25a\s+KWG",
        r"\bMaRisk\b",
        r"Risikotragfähigkeit",
        r"Gesamtbanksteuerung",
        r"Stresstests?",
        r"Interne\s+Revision",
    ],
    "wphg": [
        r"§\s*\d+\s+WpHG",
        r"\bWpHG\b",
        r"\bMaComp\b",
        r"\bMiFID\b",
        r"\bMAR\b",
        r"Interessenkonflikt",
        r"Wohlverhaltens",
        r"Anlageberatung",
    ],
}

# Regex für das Extrahieren von Rechtszitaten aus Befundtexten
_CITATION_PATTERNS = [
    re.compile(r"§\s*\d+[a-z]?(?:\s+Abs\.\s*\d+)?(?:\s+\w+){0,3}"),
    re.compile(r"Art\.\s*\d+[a-z]?(?:\s+Abs\.\s*\d+)?(?:\s+\w+){0,3}"),
    re.compile(r"MaRisk\s+[A-Z]{2}\s+\d+(?:\.\d+)*"),
]


def _extract_citations(text: str) -> list[str]:
    """Extrahiert alle Rechtszitate (§ X, Art. Y, MaRisk XY) aus einem Text."""
    citations = []
    for pattern in _CITATION_PATTERNS:
        for match in pattern.finditer(text):
            citation = match.group(0).strip()
            if citation:
                citations.append(citation)
    # Deduplizieren, Reihenfolge beibehalten
    seen: set[str] = set()
    result = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result


def _citation_in_chunks(citation: str, chunks: list[str]) -> bool:
    """Prüft, ob ein Zitat in mindestens einem der Retrieval-Chunks vorkommt.

    Neben dem exakten Match wird auch geprüft, ob der Kern des Zitats
    (Paragraphenzeichen/Art. + Nummer + erstes Wort, z.B. '§ 15 GwG')
    im Chunk enthalten ist. Das verhindert False Positives durch trailing
    Kontextwörter, die der Regex miterfasst.
    """
    citation_normalized = re.sub(r"\s+", " ", citation).strip()
    # Kern-Zitat: maximal die ersten 3 Tokens (z.B. "§ 15 GwG")
    citation_tokens = citation_normalized.split()
    citation_core = (
        " ".join(citation_tokens[:3])
        if len(citation_tokens) >= 3
        else citation_normalized
    )

    for chunk in chunks:
        chunk_normalized = re.sub(r"\s+", " ", chunk)
        if citation_normalized in chunk_normalized:
            return True
        # Fallback: Kern-Zitat im Chunk enthalten?
        if citation_core and citation_core in chunk_normalized:
            return True
    return False


class TermDriftChecker:
    """
    Post-Processing-Layer zur Erkennung von Terminologie-Drift in Befunden.

    Prüft:
    1. Phantom-Zitate: Rechtsnormen im Befund, die in keinem Retrieval-Chunk auftauchen
    2. (Erweiterbar) Verwendung generischer Paraphrasen statt spezifischer Rechtsterminologie
    """

    TERM_PATTERNS = TERM_PATTERNS

    def check_befund(
        self,
        befund_text: str,
        regulatorik: str,
        retrieved_chunks: list[str],
    ) -> list[str]:
        """
        Gibt eine Liste von Drift-Warnungen zurück.

        Args:
            befund_text: Der vollständige Text des generierten Befunds
                         (begruendung + belegte_textstellen + mangel_text).
            regulatorik: Regulatorik-Schlüssel (z.B. "gwg", "dora", "marisk", "wphg").
            retrieved_chunks: Liste der für dieses Prüffeld abgerufenen Retrieval-Chunk-Texte.

        Returns:
            Liste von Warnungs-Strings. Leer = kein Drift erkannt.
        """
        if not befund_text or not befund_text.strip():
            return []

        warnings: list[str] = []

        # Phantom-Zitat-Prüfung
        citations = _extract_citations(befund_text)
        for citation in citations:
            if not _citation_in_chunks(citation, retrieved_chunks):
                warnings.append(
                    f"🌊 Phantom-Zitat: '{citation}' erscheint im Befund, "
                    f"aber in keinem Retrieval-Chunk"
                )

        return warnings
