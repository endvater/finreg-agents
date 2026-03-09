import pytest

from ingestion.parser import RegulatoryParser


@pytest.fixture
def parser():
    return RegulatoryParser(fallback_chunk_size=500)


def test_chapter_parsing(parser):
    text = "Kapitel 1: Allgemeine Bestimmungen\nDas ist der Text."
    nodes = parser.parse_text(text, {"source": "test"})
    assert len(nodes) == 1
    assert nodes[0].metadata["hierarchy_level"] == "chapter"
    assert nodes[0].metadata["structural_id"] == "Kapitel 1"


def test_gwg_article_paragraph(parser):
    text = "Art. 1 Gegenstand\nDies ist Art 1.\n§ 1 Begriffsbestimmungen\nUnd hier § 1.\n(1) Absatz 1 Text\n(2) Absatz 2 Text"
    nodes = parser.parse_text(text, {"source": "gwg"})

    assert len(nodes) == 4

    # Check Art 1
    assert nodes[0].metadata["hierarchy_level"] == "article"
    assert nodes[0].metadata["structural_id"] == "Art. 1"

    # Check § 1
    assert nodes[1].metadata["hierarchy_level"] == "paragraph"
    assert nodes[1].metadata["structural_id"] == "§ 1"
    assert nodes[1].metadata["context_article"] == "1"

    # Check (1)
    assert nodes[2].metadata["hierarchy_level"] == "sub_paragraph"
    assert nodes[2].metadata["structural_id"] == "(1)"
    assert nodes[2].metadata["regulatory_reference"] == "Art. 1, § 1, Abs. 1"

    # Check (2)
    assert nodes[3].metadata["structural_id"] == "(2)"
    assert nodes[3].metadata["regulatory_reference"] == "Art. 1, § 1, Abs. 2"


def test_marisk_modules(parser):
    text = "Modul AT 7.3\nDas ist Text für AT 7.3.\nTz. 1 Erster Punkt\nTz. 2 Zweiter Punkt"
    nodes = parser.parse_text(text, {"source": "marisk"})

    assert len(nodes) == 3

    # Module Check (Bug 3 Fix Verify)
    assert nodes[0].metadata["hierarchy_level"] == "module"
    assert nodes[0].metadata["structural_id"] == "Modul AT 7.3"
    assert nodes[0].metadata["regulatory_reference"] == "Modul AT 7.3"

    # Tz Check
    assert nodes[1].metadata["hierarchy_level"] == "margin_no"
    assert nodes[1].metadata["structural_id"] == "Tz. 1"
    assert nodes[1].metadata["regulatory_reference"] == "Modul AT 7.3, Tz. 1"


def test_dora_article_recital(parser):
    text = "Recital 15\nThis is recital 15.\nArticle 21\nResilience testing.\n(1) First paragraph."
    nodes = parser.parse_text(text, {"source": "dora"})

    assert len(nodes) == 3
    assert nodes[0].metadata["structural_id"] == "Recital 15"
    assert nodes[1].metadata["structural_id"] == "Article 21"
    assert nodes[2].metadata["regulatory_reference"] == "Art. 21, Abs. 1"


def test_fallback_splitting(parser):
    # Text without any structural markers but longer than chunk size
    # By default, chunk size for tests is 500, so 600 chars should split
    text = "Dies ist ein sehr langer Textbaustein ohne Strukturmarker. " * 200
    nodes = parser.parse_text(text, {"source": "fallback"})

    # Should fall back to SentenceSplitter and return multiple nodes
    assert len(nodes) >= 2
    # Ensure no hierarchy was accidently assigned
    assert "hierarchy_level" not in nodes[0].metadata
