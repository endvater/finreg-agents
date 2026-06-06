"""Tests für die Dokument-Generatoren der Security-/Chaos-Eval-Sets."""

from pathlib import Path

from governance import eval_docs, eval_sets


def test_generate_all_creates_manifest_and_docs(tmp_path):
    m = eval_docs.generate_all(tmp_path)
    assert len(m["security"]) == len(eval_sets.load_security_set())
    assert len(m["chaos"]) == len(eval_sets.load_chaos_set())
    assert (tmp_path / "manifest.json").exists()
    # Jeder Security-Fall hat ein Dokument im interviews/-Ordner
    for rec in m["security"]:
        assert rec["doc_path"] and Path(rec["doc_path"]).exists()
        assert "/interviews/" in rec["doc_path"]


def test_prompt_injection_doc_embeds_attack(tmp_path):
    eval_docs.generate_all(tmp_path)
    cases = {c.case_id: c for c in eval_sets.load_security_set()}
    inj = [c for c in cases.values() if c.attack_type == "prompt_injection"][0]
    doc = tmp_path / "security" / inj.case_id / "interviews" / f"{inj.case_id}.txt"
    assert inj.injected_text in doc.read_text(encoding="utf-8")


def test_phantom_citation_doc_omits_norm(tmp_path):
    eval_docs.generate_all(tmp_path)
    cases = {c.case_id: c for c in eval_sets.load_security_set()}
    ph = [c for c in cases.values() if c.attack_type == "phantom_citation"][0]
    text = (tmp_path / "security" / ph.case_id / "interviews" / f"{ph.case_id}.txt").read_text()
    # Norm bewusst NICHT im Dokument → Modell darf sie nicht „belegt" zitieren
    assert "Art. 33" not in text


def test_missing_document_has_no_file(tmp_path):
    m = eval_docs.generate_all(tmp_path)
    missing = [r for r in m["chaos"] if r["chaos_type"] == "missing_document"]
    assert missing and missing[0]["doc_path"] is None
    # interviews/-Ordner existiert, aber leer
    inv = Path(missing[0]["input_dir"]) / "interviews"
    assert inv.exists() and not list(inv.glob("*"))


def test_oversized_doc_is_large(tmp_path):
    m = eval_docs.generate_all(tmp_path)
    big = [r for r in m["chaos"] if r["chaos_type"] == "oversized_input"][0]
    assert Path(big["doc_path"]).stat().st_size > 200_000


def test_contradictory_numbers_is_csv_in_logs(tmp_path):
    m = eval_docs.generate_all(tmp_path)
    cn = [r for r in m["chaos"] if r["chaos_type"] == "contradictory_numbers"][0]
    assert cn["doc_path"].endswith(".csv") and "/logs/" in cn["doc_path"]


def test_score_case_pass_and_fail():
    case = [c for c in eval_sets.load_chaos_set() if c.chaos_type == "corrupted_ocr"][0]
    exp = case.expected.model_dump(exclude_none=True)
    ok = eval_docs.score_case(
        {"bewertung": "nicht_prüfbar", "review_erforderlich": True}, exp)
    assert ok["passed"] is True
    bad = eval_docs.score_case(
        {"bewertung": "konform", "review_erforderlich": False}, exp)
    assert bad["passed"] is False
