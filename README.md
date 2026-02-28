# FinRegAgents v2 🏦🤖

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![Claude](https://img.shields.io/badge/Powered_by-Claude-d97706)](https://anthropic.com)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)](#)

> **AI Agent Framework für regulatorische Prüfungen** – GwG, MaRisk, DORA, WpHG/MaComp.

FinRegAgents simuliert behördliche Sonderprüfungen durch spezialisierte KI-Agenten.
Jeder Agent arbeitet einen regulatorischen Prüfkatalog gegen deine Dokumente ab und
generiert einen formellen Prüfbericht – so wie es ein BaFin- oder AMLA-Prüfer tut.

---

## Was ist neu in v2?

Version 2 ist eine vollständige Überarbeitung basierend auf einem Code-Review, das fünf kritische Architektur-Schwächen adressiert:

| Problem in v1 | Lösung in v2 |
|---|---|
| Keine Verifikationsschicht – Halluzinationen landen ungeprüft im Bericht | **Retrieval-Quality-Gate** + **Strukturelle Validierung** + **Confidence-Scoring** |
| System-Prompt ist GwG-hardcoded – DORA wird von einem "GwG-Prüfer" bewertet | **Regulatorik-spezifische System-Prompts** für jede der 4 Regulatoriken |
| `nicht_prüfbar` wird ignoriert – 80% nicht prüfbar = "KONFORM" | **Evidenz-Warnungen**: Ab 30% nicht prüfbar wird die Gesamtbewertung eingeschränkt |
| XSS im HTML-Report – Befund-Texte ungefiltert eingebettet | **html.escape()** für alle dynamischen Inhalte |
| Kein Audit-Trail, kein Checkpoint | **Audit-Trail** (Modell, Katalog-Version) + **Checkpoint** nach jeder Sektion |

### Weitere Verbesserungen

- **Confidence-Score** (0.0–1.0) pro Befund aus vier Signalen: Retrieval-Score, Evidenz-Coverage, Type-Match, LLM-Self-Assessment
- **Review-Markierung**: Befunde unter dem Confidence-Threshold werden als "Review erforderlich" markiert
- **Sektions-Eskalation**: Wenn >30% der Befunde einer Sektion Review erfordern → Warnung
- **Interview-Parsing**: Unterstützt jetzt beide JSON-Formate (Array und Dict mit `fragen_antworten`)
- **Screenshot-Memory-Fix**: base64-Daten werden nicht mehr in den Index geladen
- **Chunk-Size**: Von 512 auf 1024 Tokens erhöht (besser für regulatorische Texte)
- **Deduplizierung**: Identische Dateien in verschiedenen Ordnern werden nur einmal indexiert
- **YAML-Support**: Interview-Fragebögen in YAML werden korrekt geparst
- **Model-Default**: Sonnet statt Opus (kosteneffizient, Opus optional per `--model`)
- **Test-Suite**: Pytest-Tests für Confidence, Validierung, JSON-Parsing, Katalog-Struktur
- **Keine globale LlamaIndex-State-Mutation** mehr

---

## Architektur

```
finreg-agents/
│
├── pipeline.py              ← Hauptorchestrator (CLI + Python API)
│
├── catalog/
│   ├── gwg_catalog.json     ← GwG-Prüfkatalog (34 Prüffelder, 8 Sektionen)
│   ├── dora_catalog.json    ← DORA-Katalog (18 Prüffelder, 5 Sektionen)
│   ├── marisk_catalog.json  ← MaRisk-Katalog (22 Prüffelder, 8 Sektionen)
│   └── wphg_catalog.json    ← WpHG/MaComp-Katalog (20 Prüffelder, 7 Sektionen)
│
├── ingestion/
│   ├── ingestor.py          ← Multi-Modal Document Ingestor
│   └── interviews/          ← Beispiel-Fragebögen
│
├── agents/
│   └── pruef_agent.py       ← RAG + LLM Prüfer-Agent + Validierung + Confidence
│
├── reports/
│   └── bericht_generator.py ← Prüfbericht (JSON / MD / HTML) mit Audit-Trail
│
└── tests/
    └── test_core.py         ← Pytest-Tests für Kernkomponenten
```

### Datenfluss

```
Dokumente (PDF, Excel, Interview, Screenshot, Log)
        │
        ▼
  [GwGIngestor]              Multi-Modal Ingestion, Chunking, Dedup
        │
        ▼
  [VectorStoreIndex]         LlamaIndex + OpenAI Embeddings
        │
        ▼
  [Prüfkatalog]              94 Prüffelder in 4 Regulatoriken
        │
        │   für jedes Prüffeld:
        ▼
  [PrueferAgent]
   ├─ RAG-Retrieval          → Top-k relevante Chunks holen
   ├─ Quality-Gate       NEU → Score < Threshold? → nicht_prüfbar (kein LLM-Call)
   ├─ LLM-Bewertung          → Regulatorik-spezifischer Prompt → Claude
   ├─ Strukturelle Valid. NEU → Quellen-Cross-Check, Platzhalter, Konsistenz
   └─ Confidence-Score   NEU → 4 Signale → Score + Review-Markierung
        │
        ▼
  [Checkpoint]           NEU → Zwischenergebnis nach jeder Sektion
        │
        ▼
  [BerichtGenerator]
   ├─ JSON + Markdown + HTML
   ├─ Confidence-Bars    NEU → Visuelle Confidence-Indikatoren
   ├─ Evidenz-Warnungen  NEU → Warnung bei hohem nicht_prüfbar-Anteil
   └─ Audit-Trail        NEU → Modell, Katalog-Version, Zeitstempel
```

---

## Unterstützte Regulatorik

| Regulatorik | Sektionen | Prüffelder | Rechtsgrundlage |
|---|---|---|---|
| **GwG / AML** | 8 | 34 | GwG, §25h KWG, BaFin AuA |
| **DORA** | 5 | 18 | DORA Art. 5-46, RTS |
| **MaRisk** | 8 | 22 | MaRisk AT/BT, §25a KWG |
| **WpHG / MaComp** | 7 | 20 | WpHG, MaComp, MAR, MiFID II |

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/endvater/finreg-agents.git
cd finreg-agents
pip install -r requirements.txt
```

### 2. API-Keys setzen

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."        # für Embeddings (text-embedding-3-small)
```

### 3. Dokumente ablegen

```
meine_dokumente/
  pdfs/           → Policies, Verfahrensanweisungen, Prüfberichte (*.pdf)
  excel/          → Alert-Statistiken, Schulungsnachweise (*.xlsx, *.csv)
  interviews/     → Befragungsbögen (*.json, *.yaml)
  screenshots/    → TM-System, goAML, KYC-Oberfläche (*.png, *.jpg)
  logs/           → Systemlogs, Auditlogs (*.txt, *.log)
```

### 4. Prüfung starten

```bash
# GwG-Sonderprüfung (AML) – Default: Sonnet (kosteneffizient)
python pipeline.py --input ./docs --institution "Musterbank AG" --regulatorik gwg

# DORA-Prüfung (nur Drittparteienrisiko)
python pipeline.py --input ./docs --regulatorik dora --sektionen D04

# MaRisk-Vollprüfung mit Opus (höchste Qualität)
python pipeline.py --input ./docs --regulatorik marisk --model claude-opus-4-5

# WpHG / MaComp
python pipeline.py --input ./docs --regulatorik wphg --sektionen W02 W03 W04
```

### 5. Python API

```python
from pipeline import AuditPipeline

pipeline = AuditPipeline(
    input_dir="./meine_dokumente",
    institution="Musterbank AG",
    regulatorik="dora",
    sektionen_filter=["D01", "D02"],   # optional: Teilprüfung
    model="claude-sonnet-4-5-20250514", # optional: Modellwahl
)
report_paths = pipeline.run()
# → {"json": "...", "markdown": "...", "html": "..."}
```

### 6. Tests ausführen

```bash
pytest tests/ -v
```

---

## Confidence-Scoring

Jeder Befund erhält einen Confidence-Score (0.0–1.0), der aus vier Signalen berechnet wird:

| Signal | Gewichtung | Beschreibung |
|---|---|---|
| Retrieval-Score | 30% | Durchschnittliche Relevanz der gefundenen Chunks |
| Evidenz-Coverage | 30% | Anteil der erwarteten Evidenz, die gefunden wurde |
| Type-Match | 20% | Stimmen die Dokumenttypen (PDF, Excel, etc.) überein? |
| LLM-Self-Assessment | 20% | Selbsteinschätzung des Modells |

### Schwellenwerte

| Confidence | Aktion |
|---|---|
| < 0.40 | Automatisch `nicht_prüfbar` – LLM-Bewertung wird überschrieben |
| 0.40 – 0.70 | Befund markiert als **Review erforderlich** 🔍 |
| > 0.70 | Befund geht in den Bericht |
| >30% Review in einer Sektion | **Sektions-Eskalation** empfohlen |

---

## Strukturelle Validierung

Vor der Aufnahme in den Bericht durchläuft jeder Befund automatische Checks:

- **Quellen-Cross-Check**: Zitiert der Agent Quellen, die nicht im Retrieval waren? → Phantom-Quellen-Warnung
- **Platzhalter-Check**: Unaufgelöste `{}`-Platzhalter in Begründungen oder Mangel-Texten
- **Konsistenz-Check**: `konform` ohne Textstellen? `nicht_konform` ohne Mangel-Text?
- **Bewertungs-Konsistenz**: Mangel-Text bei `konform`-Bewertung?

Alle Warnungen werden im Befund gespeichert und im Bericht angezeigt.

---

## Bewertungsskala

| Bewertung | Bedeutung |
|---|---|
| ✅ **konform** | Anforderung vollständig erfüllt, Evidenz vorhanden |
| ⚠️ **teilkonform** | Anforderung teilweise erfüllt, Nachbesserung erforderlich |
| 🔴 **nicht_konform** | Anforderung nicht erfüllt – Mangel im Bericht |
| ❓ **nicht_prüfbar** | Keine ausreichende Evidenz im Prüfungskorpus |

**Schweregrade:** `wesentlich` (sofortiger Handlungsbedarf) · `bedeutsam` · `gering`

### Gesamtbewertungslogik (v2)

| Bedingung | Gesamtbewertung |
|---|---|
| Wesentliche Mängel vorhanden | **ERHEBLICHE MÄNGEL** |
| ≥50% nicht prüfbar | **UNZUREICHENDE EVIDENZ – PRÜFUNG NICHT BELASTBAR** |
| Mängel oder ≥3 teilkonform | **MÄNGEL FESTGESTELLT** |
| ≥30% nicht prüfbar | **EINGESCHRÄNKT BELASTBAR** |
| Teilkonforme Befunde vorhanden | **TEILKONFORM – NACHBESSERUNG ERFORDERLICH** |
| Alles konform | **KONFORM** |

---

## Eigenen Katalog erstellen

Jedes Prüffeld folgt diesem Schema:

```json
{
  "katalog_version": "2025-01",
  "basis": ["MaRisk 2023", "BaFin-Rundschreiben"],
  "pruefsektionen": [
    {
      "id": "S01",
      "titel": "Interne Kontrollsysteme",
      "rechtsgrundlagen": ["MaRisk AT 4.3"],
      "prueffelder": [
        {
          "id": "S01-01",
          "frage": "Ist ein IKS dokumentiert und implementiert?",
          "erwartete_evidenz": ["IKS-Dokumentation", "Prozesshandbuch"],
          "input_typen": ["pdf", "interview"],
          "bewertungskriterien": "Schriftliche IKS-Dokumentation muss vorliegen",
          "schweregrad": "wesentlich",
          "mangel_template": "Ein dokumentiertes IKS gemäß MaRisk AT 4.3 fehlt."
        }
      ]
    }
  ]
}
```

```bash
python pipeline.py --input ./docs --catalog ./mein_katalog.json
```

---

## Interview-Format

Strukturierte Befragungsprotokolle werden direkt in den Index aufgenommen.
Unterstützt werden zwei JSON-Formate:

**Format A – Dict mit Metadaten (empfohlen):**

```json
{
  "meta": {
    "institut": "Musterbank AG",
    "datum": "2025-02-01",
    "interviewer": "Prüfer KI"
  },
  "fragen_antworten": [
    {
      "id": "I-01",
      "prueffeld_referenz": "S04-01",
      "frage": "Seit wann sind Sie als GwB bestellt?",
      "antwort": "Seit März 2022, BaFin-Meldung am 20.03.2022.",
      "kommentar": "Bestellungsbeschluss liegt vor."
    }
  ]
}
```

**Format B – Einfaches Array:**

```json
[
  {"frage": "...", "antwort": "...", "kommentar": "..."}
]
```

---

## Prüfbericht-Output

Jede Prüfung erzeugt drei Dateien:

| Format | Verwendung |
|---|---|
| **JSON** | Maschinenlesbar, API-Integration, Weiterverarbeitung |
| **Markdown** | Lesbar, Git-kompatibel, Review-Workflows |
| **HTML** | Druckfähig, Präsentation, PDF-Konvertierung |

Alle Berichte enthalten jetzt:
- Confidence-Bars pro Befund
- Review-Markierungen (🔍) für unsichere Bewertungen
- Validierungshinweise (⚡) bei strukturellen Problemen
- Evidenz-Warnungen bei hohem nicht_prüfbar-Anteil
- Audit-Trail mit Modell, Katalog-Version und Zeitstempel

---

## Kosten-Einschätzung

| Regulatorik | Prüffelder | Geschätzter Aufwand (Sonnet) | Geschätzter Aufwand (Opus) |
|---|---|---|---|
| GwG | 34 | ~$0.80–1.50 | ~$8–15 |
| DORA | 18 | ~$0.40–0.80 | ~$4–8 |
| MaRisk | 22 | ~$0.50–1.00 | ~$5–10 |
| WpHG | 20 | ~$0.45–0.90 | ~$4.50–9 |

Hinweis: Kosten hängen von Dokumentenmenge, Chunk-Anzahl und Antwortlänge ab. Durch das Retrieval-Quality-Gate in v2 werden unnötige LLM-Calls bei schlechtem Retrieval eingespart.

---

## Roadmap

- [ ] Skeptiker-Agent: Adversariales LLM-Review als optionaler Post-Processing-Layer
- [ ] Synthetische Kontroll-Prüffelder (Ground-Truth-Signal) zur Kalibrierung
- [ ] Persistenter Vektorindex via ChromaDB / Weaviate
- [ ] Claude Vision für Screenshot-Analyse (TM-Systeme, KYC-Oberflächen)
- [ ] Delta-Prüfung – nur geänderte Dokumente neu einlesen
- [ ] Streamlit-UI für interaktive Prüfung mit Sampling-Audit
- [ ] JSON-Schema für Custom-Kataloge mit Validierung beim Laden
- [ ] Multi-Institut-Vergleich – Benchmarking über Institutsgrenzen

---

## Disclaimer

FinRegAgents ist ein **Simulations- und Vorbereitungstool**. Es ersetzt **keine
offizielle BaFin-Prüfung** und begründet keine Rechtsberatung. Prüfungsergebnisse
sind als interne Vorbereitung zu verstehen, nicht als behördliche Feststellung.

Die Confidence-Scores und Review-Markierungen dienen dazu, die Belastbarkeit
der einzelnen Befunde transparent zu machen. Befunde mit niedrigem Confidence
oder Review-Markierung sollten stets manuell validiert werden.

---

## Contributing

Contributions willkommen – insbesondere:

- Neue Prüfkataloge für weitere Regulatoriken
- Verbesserte Prüffragen und Bewertungskriterien
- Skeptiker-Agent-Implementierung
- Neue Ingestion-Adapter (z.B. .docx, Notion, Confluence)
- Tests und Benchmarks

Bitte fork → branch → PR mit Beschreibung welche Regulatorik / welches Feature erweitert wurde.

---

## Lizenz

Apache License 2.0 – siehe [LICENSE](LICENSE).

Du kannst FinRegAgents frei nutzen, modifizieren und in kommerzielle Produkte
integrieren, solange der Copyright-Vermerk erhalten bleibt.

---

Gebaut mit LlamaIndex · LangChain · Claude · ❤️
