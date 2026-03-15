import logging

from ui_drift import build_befund_index, build_drift_rows


def test_build_befund_index_warns_on_duplicate_id(caplog):
    payload = {
        "sektionen": [
            {"id": "S1", "befunde": [{"id": "PF-1", "bewertung": "konform"}]},
            {"id": "S2", "befunde": [{"id": "PF-1", "bewertung": "nicht_konform"}]},
        ]
    }
    with caplog.at_level(logging.WARNING):
        index = build_befund_index(payload)

    assert index["PF-1"]["sektion"] == "S2"
    assert "Doppelte prueffeld_id 'PF-1'" in caplog.text


def test_build_drift_rows_maps_improved_and_worsened():
    index_a = {
        "PF-1": {"bewertung": "nicht_konform", "confidence": 0.5},
        "PF-2": {"bewertung": "konform", "confidence": 0.7},
    }
    index_b = {
        "PF-1": {"bewertung": "konform", "confidence": 0.8},
        "PF-2": {"bewertung": "teilkonform", "confidence": 0.6},
    }

    rows = {row["prueffeld_id"]: row for row in build_drift_rows(index_a, index_b)}
    assert rows["PF-1"]["status"] == "verbessert"
    assert rows["PF-2"]["status"] == "verschlechtert"


def test_build_drift_rows_handles_new_and_removed():
    index_a = {"PF-1": {"bewertung": "konform", "confidence": 0.9}}
    index_b = {"PF-2": {"bewertung": "konform", "confidence": 0.8}}

    rows = {row["prueffeld_id"]: row for row in build_drift_rows(index_a, index_b)}
    assert rows["PF-1"]["status"] == "entfallen"
    assert rows["PF-2"]["status"] == "neu"


def test_build_drift_rows_treats_disputed_and_nicht_pruefbar_as_equal():
    index_a = {"PF-1": {"bewertung": "nicht_prüfbar", "confidence": 0.2}}
    index_b = {"PF-1": {"bewertung": "disputed", "confidence": 0.2}}

    rows = build_drift_rows(index_a, index_b)
    assert rows[0]["status"] == "gleich"


def test_build_drift_rows_warns_on_unknown_bewertung(caplog):
    index_a = {"PF-1": {"bewertung": "konform", "confidence": 0.2}}
    index_b = {"PF-1": {"bewertung": "future_state", "confidence": 0.2}}

    with caplog.at_level(logging.WARNING):
        rows = build_drift_rows(index_a, index_b)

    assert rows[0]["status"] == "geändert"
    assert "Unbekannte Bewertung 'future_state' für prueffeld 'PF-1'" in caplog.text
