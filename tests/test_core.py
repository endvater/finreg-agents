"""
Tests für FinRegAgents v2 – Kernkomponenten
Ausführen: pytest tests/ -v
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# ------------------------------------------------------------------ #
# Test: Confidence-Berechnung
# ------------------------------------------------------------------ #

from agents.pruef_agent import (
    compute_confidence,
    extract_json,
    validate_befund_structure,
    Bewertung,
    Sektionsergebnis,
    Befund,
)


class TestConfidenceScore:

    def test_perfect_confidence(self):
        score = compute_confidence(
            retrieval_scores=[0.95, 0.90, 0.85],
            erwartete_evidenz=["Risikoanalyse", "Versionierung"],
            gefundene_quellen=["risikoanalyse_2024.pdf", "versionierung.xlsx"],
            erlaubte_typen={"pdf", "excel"},
            gefundene_typen={"pdf", "excel"},
            llm_confidence=0.9,
        )
        assert 0.7 < score <= 1.0

    def test_zero_retrieval(self):
        score = compute_confidence(
            retrieval_scores=[],
            erwartete_evidenz=["Risikoanalyse"],
            gefundene_quellen=[],
            erlaubte_typen={"pdf"},
            gefundene_typen=set(),
            llm_confidence=0.5,
        )
        assert score < 0.3

    def test_low_retrieval_scores(self):
        score = compute_confidence(
            retrieval_scores=[0.1, 0.15, 0.2],
            erwartete_evidenz=["Risikoanalyse"],
            gefundene_quellen=["unrelated.pdf"],
            erlaubte_typen={"pdf"},
            gefundene_typen={"pdf"},
            llm_confidence=0.8,
        )
        # Low retrieval should pull down even with good LLM confidence
        assert score < 0.6

    def test_no_type_restriction(self):
        """Keine input_typen im Prüffeld → type_signal = 1.0"""
        score = compute_confidence(
            retrieval_scores=[0.8],
            erwartete_evidenz=[],
            gefundene_quellen=["doc.pdf"],
            erlaubte_typen=set(),
            gefundene_typen={"pdf"},
            llm_confidence=0.7,
        )
        assert score > 0.5

    def test_confidence_bounded(self):
        """Score bleibt zwischen 0 und 1."""
        score = compute_confidence(
            retrieval_scores=[1.5, 2.0],  # Unrealistisch hoch
            erwartete_evidenz=[],
            gefundene_quellen=[],
            erlaubte_typen=set(),
            gefundene_typen=set(),
            llm_confidence=1.5,
        )
        assert 0.0 <= score <= 1.0


# ------------------------------------------------------------------ #
# Test: JSON-Extraktion
# ------------------------------------------------------------------ #

class TestJsonExtraction:

    def test_clean_json(self):
        result = extract_json('{"bewertung": "konform", "begruendung": "Alles gut"}')
        assert result["bewertung"] == "konform"

    def test_markdown_wrapped(self):
        raw = '```json\n{"bewertung": "teilkonform"}\n```'
        result = extract_json(raw)
        assert result["bewertung"] == "teilkonform"

    def test_with_preamble(self):
        raw = 'Hier ist meine Bewertung:\n\n{"bewertung": "nicht_konform", "begruendung": "Fehlt"}'
        result = extract_json(raw)
        assert result["bewertung"] == "nicht_konform"

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            extract_json("Das ist kein JSON")


# ------------------------------------------------------------------ #
# Test: Strukturelle Validierung
# ------------------------------------------------------------------ #

class TestStructuralValidation:

    def test_phantom_quellen(self):
        warnings = validate_befund_structure(
            llm_result={
                "bewertung": "konform",
                "quellen": ["risikoanalyse.pdf", "phantom.docx"],
                "belegte_textstellen": ["Zitat"],
            },
            retrieved_sources={"risikoanalyse.pdf"},
            regulatorik="gwg",
        )
        assert any("Phantom-Quellen" in w for w in warnings)

    def test_konform_without_textstellen(self):
        warnings = validate_befund_structure(
            llm_result={
                "bewertung": "konform",
                "quellen": ["doc.pdf"],
                "belegte_textstellen": [],
            },
            retrieved_sources={"doc.pdf"},
            regulatorik="gwg",
        )
        assert any("ohne belegte Textstellen" in w for w in warnings)

    def test_konform_with_mangel(self):
        warnings = validate_befund_structure(
            llm_result={
                "bewertung": "konform",
                "quellen": ["doc.pdf"],
                "belegte_textstellen": ["Zitat"],
                "mangel_text": "Ein Mangel existiert",
            },
            retrieved_sources={"doc.pdf"},
            regulatorik="gwg",
        )
        assert any("Mangel-Text bei 'konform'" in w for w in warnings)

    def test_nicht_konform_ohne_mangel(self):
        warnings = validate_befund_structure(
            llm_result={
                "bewertung": "nicht_konform",
                "quellen": ["doc.pdf"],
                "belegte_textstellen": ["Zitat"],
                "mangel_text": None,
            },
            retrieved_sources={"doc.pdf"},
            regulatorik="gwg",
        )
        assert any("Kein Mangel-Text bei 'nicht_konform'" in w for w in warnings)

    def test_clean_befund_no_warnings(self):
        warnings = validate_befund_structure(
            llm_result={
                "bewertung": "konform",
                "quellen": ["doc.pdf"],
                "belegte_textstellen": ["Beleg aus doc.pdf"],
            },
            retrieved_sources={"doc.pdf"},
            regulatorik="gwg",
        )
        assert warnings == []

    def test_placeholder_detection(self):
        warnings = validate_befund_structure(
            llm_result={
                "bewertung": "nicht_konform",
                "quellen": [],
                "belegte_textstellen": [],
                "begruendung": "Gemäß {paragraph} fehlt die Dokumentation",
                "mangel_text": "Mangel vorhanden",
            },
            retrieved_sources=set(),
            regulatorik="gwg",
        )
        assert any("Platzhalter" in w for w in warnings)


# ------------------------------------------------------------------ #
# Test: Sektionsergebnis
# ------------------------------------------------------------------ #

class TestSektionsergebnis:

    def test_review_quote(self):
        s = Sektionsergebnis(sektion_id="S01", titel="Test")
        s.befunde = [
            Befund(prueffeld_id="S01-01", frage="?", bewertung=Bewertung.KONFORM,
                   begruendung="ok", review_erforderlich=True),
            Befund(prueffeld_id="S01-02", frage="?", bewertung=Bewertung.KONFORM,
                   begruendung="ok", review_erforderlich=False),
        ]
        assert s.review_quote == 0.5

    def test_kritische_befunde(self):
        s = Sektionsergebnis(sektion_id="S01", titel="Test")
        s.befunde = [
            Befund(prueffeld_id="S01-01", frage="?", bewertung=Bewertung.KONFORM, begruendung="ok"),
            Befund(prueffeld_id="S01-02", frage="?", bewertung=Bewertung.NICHT_KONFORM, begruendung="nicht ok"),
            Befund(prueffeld_id="S01-03", frage="?", bewertung=Bewertung.TEILKONFORM, begruendung="teils"),
        ]
        assert len(s.kritische_befunde) == 2


# ------------------------------------------------------------------ #
# Test: Interview-Ingestion
# ------------------------------------------------------------------ #

class TestInterviewIngestion:

    def test_dict_format_with_fragen_antworten(self):
        from ingestion.ingestor import GwGIngestor
        ingestor = GwGIngestor()
        data = {
            "meta": {"institut": "Testbank", "datum": "2025-01-01"},
            "fragen_antworten": [
                {"id": "I-01", "frage": "Test?", "antwort": "Ja", "kommentar": "OK"}
            ]
        }
        text = ingestor._interview_data_to_text(data, "test.json")
        assert "institut: Testbank" in text
        assert "Frage I-01" in text
        assert "Antwort: Ja" in text

    def test_array_format(self):
        from ingestion.ingestor import GwGIngestor
        ingestor = GwGIngestor()
        data = [
            {"frage": "Frage 1?", "antwort": "Antwort 1"}
        ]
        text = ingestor._interview_data_to_text(data, "test.json")
        assert "Frage 1?" in text
        assert "Antwort 1" in text


# ------------------------------------------------------------------ #
# Test: Bericht-Generator XSS-Schutz
# ------------------------------------------------------------------ #

class TestBerichtXSS:

    def test_html_escape_in_befund(self):
        from reports.bericht_generator import _esc
        dangerous = '<script>alert("xss")</script>'
        escaped = _esc(dangerous)
        assert "<script>" not in escaped
        assert "&lt;script&gt;" in escaped

    def test_none_handling(self):
        from reports.bericht_generator import _esc
        assert _esc(None) == ""


# ------------------------------------------------------------------ #
# Test: Katalog-Validierung
# ------------------------------------------------------------------ #

class TestKatalogStruktur:

    @pytest.mark.parametrize("catalog_file", [
        "catalog/gwg_catalog.json",
        "catalog/dora_catalog.json",
        "catalog/marisk_catalog.json",
        "catalog/wphg_catalog.json",
    ])
    def test_katalog_pflichtfelder(self, catalog_file):
        path = Path(__file__).parent.parent / catalog_file
        if not path.exists():
            pytest.skip(f"Katalog nicht gefunden: {path}")
        katalog = json.loads(path.read_text(encoding="utf-8"))

        assert "katalog_version" in katalog
        assert "pruefsektionen" in katalog
        assert len(katalog["pruefsektionen"]) > 0

        for sektion in katalog["pruefsektionen"]:
            assert "id" in sektion, f"Sektion ohne ID"
            assert "titel" in sektion, f"Sektion {sektion.get('id')} ohne Titel"
            assert "prueffelder" in sektion, f"Sektion {sektion['id']} ohne Prüffelder"

            for feld in sektion["prueffelder"]:
                assert "id" in feld, f"Prüffeld ohne ID in {sektion['id']}"
                assert "frage" in feld, f"Prüffeld {feld.get('id')} ohne Frage"
                assert "schweregrad" in feld, f"Prüffeld {feld['id']} ohne Schweregrad"
                assert feld["schweregrad"] in ("wesentlich", "bedeutsam", "gering"), \
                    f"Ungültiger Schweregrad in {feld['id']}: {feld['schweregrad']}"
