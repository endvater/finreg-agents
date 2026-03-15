# Changelog

Alle relevanten Änderungen an FinRegAgents werden hier dokumentiert.

## [Unreleased]

### Changed
- README Installationsabschnitt auf Python 3.12 und Provider-Extras präzisiert.
- README um Troubleshooting-Matrix für typische Setup-/Runtime-Fehler ergänzt.

### Fixed
- Streamlit: Provider-spezifische API-Key-Felder und OLLAMA-Host sauber verdrahtet.
- Embedding-Auswahl: Fallback-Logik ergänzt, wenn `fastembed` nicht installiert ist.
- Pipeline/Agent-Kompatibilität: `PrueferAgent` akzeptiert wieder das Argument `adversarial`.
- Gemini Defaults aktualisiert:
  - LLM: `gemini-3.1-pro-001`
  - Embeddings: `models/gemini-embedding-001`
- Ingestion: PDF-Chunking stabilisiert (korrekte `chunk_size`-Verwendung).
- Security/Upload-Hygiene: Upload-Dateinamen werden auf Basename normalisiert.

## [v2.2]

### Added
- Adversarial Prompting Layer als optionaler zweiter LLM-Pass mit Divergenz-Logik.
- Confidence Guards (`MIN_INPUT_TOKENS`, `MIN_DISTINCT_SOURCES`, `MIN_EVIDENCE_QUOTES`).
- Claim-Annotationen zur besseren Auditierbarkeit der Befunde.

### Fixed
- Retry-Mechanismen mit Exponential Backoff für robuste LLM-Aufrufe.
- Type-Scoping vor Retrieval-Gate und Confidence-Berechnung (kein Input-Typ-Leak).
- Fuzzy-Matching für Evidenz-Coverage auf Token-Grenzen korrigiert.
- Regulatorik-Validierung: Plausibilitätscheck für Rechtszitate aktiviert.
- Logging konsolidiert (`logging` statt verstreuter `print`-Ausgaben).

## [v2.1]

### Added
- Skeptiker-Agent als optionaler adversarialer Post-Processing-Layer.
- CI-Absicherung mit Tests + Ruff-Linting.

### Fixed
- `.env`-Laden beim Start (`load_dotenv`) wiederhergestellt.
- Fehlertoleranz bei Checkpointing verbessert (Warn-Logging statt stummem `except`).
- Mehrere Stabilitätsfixes in JSON-Parsing, Typnormalisierung und Bewertungsfluss.
