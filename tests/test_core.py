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
    confidence_level_from_score,
    evaluate_confidence_guards,
    extract_json,
    validate_befund_structure,
    PrueferAgent,
    Bewertung,
    Sektionsergebnis,
    Befund,
)
from agents.skeptiker_agent import (
    SkeptikerAgent,
    SkeptikerBefund,
    merge_befund_skeptiker,
    SKEPTIKER_MIN_CONFIDENCE,
    SKEPTIKER_CONFIDENCE_PENALTY,
)
from pipeline import AuditPipeline


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

    def test_confidence_level_mapping(self):
        assert confidence_level_from_score(0.85) == "high"
        assert confidence_level_from_score(0.6) == "medium"
        assert confidence_level_from_score(0.2) == "low"

    def test_confidence_guards_fail_and_pass(self):
        failed = evaluate_confidence_guards(
            input_tokens=120, distinct_sources=1, evidence_quotes=0
        )
        assert failed["passed"] is False
        assert "MIN_INPUT_TOKENS" in failed["violations"]
        assert "MIN_DISTINCT_SOURCES" in failed["violations"]
        assert "MIN_EVIDENCE_QUOTES" in failed["violations"]

        passed = evaluate_confidence_guards(
            input_tokens=420, distinct_sources=3, evidence_quotes=2
        )
        assert passed["passed"] is True
        assert passed["violations"] == []


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

    def test_unplausible_law_reference_detected(self):
        warnings = validate_befund_structure(
            llm_result={
                "bewertung": "teilkonform",
                "quellen": [],
                "belegte_textstellen": ["Prüfverweis: § 99 XYZ"],
                "begruendung": "Gemäß § 99 XYZ bestehen Defizite.",
            },
            retrieved_sources=set(),
            regulatorik="gwg",
        )
        assert any("Unplausible Rechtszitate" in w for w in warnings)

    def test_valid_gwg_reference_with_abs_not_flagged(self):
        warnings = validate_befund_structure(
            llm_result={
                "bewertung": "teilkonform",
                "quellen": [],
                "belegte_textstellen": ["Verweis auf § 15 Abs. 2 GwG ist vorhanden."],
                "begruendung": "Die Maßnahme orientiert sich an § 15 Abs. 2 GwG.",
            },
            retrieved_sources=set(),
            regulatorik="gwg",
        )
        assert not any("Unplausible Rechtszitate" in w for w in warnings)


# ------------------------------------------------------------------ #
# Test: Sektionsergebnis
# ------------------------------------------------------------------ #


class TestSektionsergebnis:
    def test_review_quote(self):
        s = Sektionsergebnis(sektion_id="S01", titel="Test")
        s.befunde = [
            Befund(
                prueffeld_id="S01-01",
                frage="?",
                bewertung=Bewertung.KONFORM,
                begruendung="ok",
                review_erforderlich=True,
            ),
            Befund(
                prueffeld_id="S01-02",
                frage="?",
                bewertung=Bewertung.KONFORM,
                begruendung="ok",
                review_erforderlich=False,
            ),
        ]
        assert s.review_quote == 0.5

    def test_kritische_befunde(self):
        s = Sektionsergebnis(sektion_id="S01", titel="Test")
        s.befunde = [
            Befund(
                prueffeld_id="S01-01",
                frage="?",
                bewertung=Bewertung.KONFORM,
                begruendung="ok",
            ),
            Befund(
                prueffeld_id="S01-02",
                frage="?",
                bewertung=Bewertung.NICHT_KONFORM,
                begruendung="nicht ok",
            ),
            Befund(
                prueffeld_id="S01-03",
                frage="?",
                bewertung=Bewertung.TEILKONFORM,
                begruendung="teils",
            ),
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
            ],
        }
        text = ingestor._interview_data_to_text(data, "test.json")
        assert "institut: Testbank" in text
        assert "Frage I-01" in text
        assert "Antwort: Ja" in text

    def test_array_format(self):
        from ingestion.ingestor import GwGIngestor

        ingestor = GwGIngestor()
        data = [{"frage": "Frage 1?", "antwort": "Antwort 1"}]
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
# Test: Skeptiker-Agent
# ------------------------------------------------------------------ #


class TestSkeptikerAgent:
    def _make_befund(self, bewertung=Bewertung.KONFORM, confidence=0.75, review=False):
        return Befund(
            prueffeld_id="S01-01",
            frage="Ist ein IKS dokumentiert?",
            bewertung=bewertung,
            begruendung="Das IKS ist vollständig dokumentiert.",
            belegte_textstellen=["IKS-Handbuch S.12: 'Kontrollrahmen ist definiert'"],
            quellen=["iks_handbuch.pdf"],
            confidence=confidence,
            review_erforderlich=review,
        )

    def _make_prueffeld(self):
        return {
            "id": "S01-01",
            "frage": "Ist ein IKS dokumentiert?",
            "erwartete_evidenz": ["IKS-Dokumentation", "Prozesshandbuch"],
            "input_typen": ["pdf"],
            "bewertungskriterien": "Schriftliche IKS-Dokumentation muss vorliegen",
            "rechtsgrundlagen": ["MaRisk AT 4.3"],
            "schweregrad": "wesentlich",
        }

    def test_pass_through_nicht_pruefbar(self):
        """Nicht-prüfbare Befunde werden nicht gesceptikert."""
        agent = SkeptikerAgent()
        befund = self._make_befund(bewertung=Bewertung.NICHT_PRUEFBAR, confidence=0.0)
        result = agent.reviewe(befund, self._make_prueffeld())
        assert result.akzeptiert is True
        assert result.adjustierter_confidence == 0.0

    def test_pass_through_low_confidence(self):
        """Zu niedrig-confidence Befunde werden übersprungen."""
        agent = SkeptikerAgent()
        befund = self._make_befund(confidence=SKEPTIKER_MIN_CONFIDENCE - 0.1)
        result = agent.reviewe(befund, self._make_prueffeld())
        assert result.akzeptiert is True
        assert result.adjustierter_confidence == befund.confidence

    def test_only_konform_skips_nicht_konform(self):
        """only_konform=True überspringt nicht_konform Befunde."""
        agent = SkeptikerAgent(only_konform=True)
        befund = self._make_befund(bewertung=Bewertung.NICHT_KONFORM, confidence=0.8)
        result = agent.reviewe(befund, self._make_prueffeld())
        assert result.akzeptiert is True

    def test_confidence_penalty_applied(self):
        """Confidence-Penalty wird bei Einwänden abgezogen."""
        befund = self._make_befund(confidence=0.8)
        skeptiker = SkeptikerBefund(
            prueffeld_id="S01-01",
            original_bewertung=Bewertung.KONFORM,
            original_confidence=0.8,
            akzeptiert=False,
            bewertung_empfehlung=Bewertung.TEILKONFORM,
            einwaende=["Evidenz zu vage", "Audit-Trail fehlt"],
            adjustierter_confidence=max(0.0, 0.8 - 2 * SKEPTIKER_CONFIDENCE_PENALTY),
        )
        merged = merge_befund_skeptiker(befund, skeptiker)
        assert merged.confidence < befund.confidence
        assert merged.review_erforderlich is True

    def test_merge_akzeptiert_keeps_original(self):
        """Bei Akzeptanz bleibt originale Bewertung und Confidence."""
        befund = self._make_befund(confidence=0.8)
        skeptiker = SkeptikerBefund(
            prueffeld_id="S01-01",
            original_bewertung=Bewertung.KONFORM,
            original_confidence=0.8,
            akzeptiert=True,
            bewertung_empfehlung=None,
            einwaende=[],
            adjustierter_confidence=0.8,
        )
        merged = merge_befund_skeptiker(befund, skeptiker)
        assert merged.bewertung == Bewertung.KONFORM
        assert merged.confidence == 0.8
        assert merged.review_erforderlich is False

    def test_merge_adds_einwand_hints(self):
        """Einwände landen in validierungshinweisen."""
        befund = self._make_befund()
        skeptiker = SkeptikerBefund(
            prueffeld_id="S01-01",
            original_bewertung=Bewertung.KONFORM,
            original_confidence=0.75,
            akzeptiert=False,
            bewertung_empfehlung=Bewertung.TEILKONFORM,
            einwaende=["Prozessnachweis fehlt"],
            fehlende_evidenz=["Audit-Trail"],
            adjustierter_confidence=0.6,
        )
        merged = merge_befund_skeptiker(befund, skeptiker)
        hints = " ".join(merged.validierungshinweise)
        assert "Skeptiker widerspricht" in hints
        assert "Prozessnachweis fehlt" in hints
        assert "Audit-Trail" in hints

    def test_skeptiker_befund_dataclass(self):
        """SkeptikerBefund kann instanziiert werden."""
        sb = SkeptikerBefund(
            prueffeld_id="T01-01",
            original_bewertung=Bewertung.KONFORM,
            original_confidence=0.7,
            akzeptiert=True,
            bewertung_empfehlung=None,
            adjustierter_confidence=0.7,
        )
        assert sb.akzeptiert is True
        assert sb.einwaende == []
        assert sb.fehlende_evidenz == []

    def test_build_skeptiker_befund_normalizes_types(self):
        befund = self._make_befund(confidence=0.8)
        agent = SkeptikerAgent.__new__(SkeptikerAgent)
        result = agent._build_skeptiker_befund(
            befund,
            {
                "akzeptiert": "false",
                "bewertung_empfehlung": "teilkonform",
                "einwaende": "Evidenz zu vage",
                "staerken": "Gute Struktur",
                "schweregrad_erhoehen": "true",
                "nachforderung_empfohlen": "false",
                "fehlende_evidenz": "Audit-Trail",
            },
        )
        assert result.akzeptiert is False
        assert result.einwaende == ["Evidenz zu vage"]
        assert result.staerken == ["Gute Struktur"]
        assert result.fehlende_evidenz == ["Audit-Trail"]
        assert result.schweregrad_erhoehen is True
        assert result.nachforderung_empfohlen is False
        assert result.adjustierter_confidence == 0.65

    def test_build_skeptiker_befund_normalizes_int_booleans(self):
        befund = self._make_befund(confidence=0.8)
        agent = SkeptikerAgent.__new__(SkeptikerAgent)
        result = agent._build_skeptiker_befund(
            befund,
            {
                "akzeptiert": 0,
                "bewertung_empfehlung": "teilkonform",
                "schweregrad_erhoehen": 1,
                "nachforderung_empfohlen": 0,
            },
        )
        assert result.akzeptiert is False
        assert result.schweregrad_erhoehen is True
        assert result.nachforderung_empfohlen is False


class _FakeNode:
    def __init__(self, score, metadata=None, content="evidenz"):
        self.score = score
        self.metadata = metadata or {}
        self._content = content

    def get_content(self):
        return self._content


class TestPrueferAgentTypeScoping:
    def test_disallowed_types_force_nicht_pruefbar(self):
        agent = PrueferAgent.__new__(PrueferAgent)
        agent.retrieval_score_min = 0.35
        agent.regulatorik = "gwg"
        agent._retrieve_evidence = MagicMock(
            return_value=[
                _FakeNode(0.95, {"input_type": "log", "source": "tm.log"}, "logcontent")
            ]
        )
        agent._evaluate_with_llm = MagicMock(
            side_effect=AssertionError("LLM darf hier nicht aufgerufen werden")
        )
        agent._format_evidence = MagicMock(return_value="unused")

        befund = agent.pruefe_feld(
            {
                "id": "S01-99",
                "frage": "Testfrage",
                "input_typen": ["pdf"],
                "schweregrad": "bedeutsam",
            }
        )
        assert befund.bewertung == Bewertung.NICHT_PRUEFBAR
        assert befund.review_erforderlich is True
        assert "erlaubten Dokumenttypen" in befund.begruendung
        assert any(
            "unzulässige Dokumenttypen" in h for h in befund.validierungshinweise
        )


# ------------------------------------------------------------------ #
# Test: Katalog-Validierung
# ------------------------------------------------------------------ #


class TestKatalogStruktur:
    @pytest.mark.parametrize(
        "catalog_file",
        [
            "catalog/gwg_catalog.json",
            "catalog/dora_catalog.json",
            "catalog/marisk_catalog.json",
            "catalog/wphg_catalog.json",
        ],
    )
    def test_katalog_pflichtfelder(self, catalog_file):
        path = Path(__file__).parent.parent / catalog_file
        if not path.exists():
            pytest.skip(f"Katalog nicht gefunden: {path}")
        katalog = json.loads(path.read_text(encoding="utf-8"))

        assert "katalog_version" in katalog
        assert "pruefsektionen" in katalog
        assert len(katalog["pruefsektionen"]) > 0

        for sektion in katalog["pruefsektionen"]:
            assert "id" in sektion, "Sektion ohne ID"
            assert "titel" in sektion, f"Sektion {sektion.get('id')} ohne Titel"
            assert "prueffelder" in sektion, f"Sektion {sektion['id']} ohne Prüffelder"

            for feld in sektion["prueffelder"]:
                assert "id" in feld, f"Prüffeld ohne ID in {sektion['id']}"
                assert "frage" in feld, f"Prüffeld {feld.get('id')} ohne Frage"
                assert "schweregrad" in feld, f"Prüffeld {feld['id']} ohne Schweregrad"
                assert feld["schweregrad"] in ("wesentlich", "bedeutsam", "gering"), (
                    f"Ungültiger Schweregrad in {feld['id']}: {feld['schweregrad']}"
                )


class TestPipelineScopeValidation:
    @patch("pipeline.BerichtGenerator")
    @patch("pipeline.PrueferAgent")
    @patch("pipeline.VectorStoreIndex")
    @patch("pipeline.OpenAIEmbedding")
    @patch("pipeline.Settings")
    @patch("pipeline.GwGIngestor")
    def test_run_raises_if_no_fields_processed(
        self,
        mock_ingestor_cls,
        mock_settings,
        _mock_embedding,
        mock_vector_store_index,
        _mock_pruefer_cls,
        _mock_bericht_cls,
    ):
        mock_ingestor = MagicMock()
        mock_ingestor.ingest_directory.return_value = [object()]
        mock_ingestor_cls.return_value = mock_ingestor
        mock_vector_store_index.from_documents.return_value = MagicMock()
        mock_settings.embed_model = None

        pipeline = AuditPipeline(
            input_dir="demo",
            regulatorik="gwg",
            sektionen_filter=["NICHT_EXISTENT"],
            verbose=False,
        )

        with pytest.raises(ValueError, match="Keine Prüffelder wurden verarbeitet"):
            pipeline.run()


class TestTokenStats:
    def test_write_run_stats_contains_required_fields(self, tmp_path):
        pipeline = AuditPipeline(
            input_dir="demo",
            output_dir=str(tmp_path),
            verbose=False,
        )
        pipeline._add_token_usage("pruefer", {"input": 1000, "output": 400, "total": 1400})
        pipeline._add_token_usage("skeptiker", {"input": 200, "output": 100, "total": 300})
        stats_file, costs = pipeline._write_run_stats()

        payload = json.loads(Path(stats_file).read_text(encoding="utf-8"))
        assert payload["token_stats"]["version"] == "1.0"
        assert payload["token_stats"]["gesamt"]["total"] == 1700
        assert payload["token_stats"]["nach_agent"]["pruefer"]["input"] == 1000
        assert "kosten_schaetzung" in payload
        assert "pricing_timestamp" in payload["kosten_schaetzung"]
        assert payload["stats_file"].endswith("run_stats.json")
