# FinRegAgents 🏦🤖

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.11%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.11%2B-ff6b35)](https://llamaindex.ai)
[![LangChain](https://img.shields.io/badge/LangChain-0.3%2B-1c3c3c)](https://langchain.com)
[![Claude](https://img.shields.io/badge/Powered_by-Claude_3-d97706)](https://anthropic.com)
[![Status](https://img.shields.io/badge/Status-Alpha-orange)]()

> **AI agent framework for financial regulatory audits** – GwG, MaRisk, DORA and beyond.

FinRegAgents simuliert behördliche Sonderprüfungen durch spezialisierte KI-Agenten.
Jeder Agent arbeitet einen regulatorischen Prüfkatalog gegen deine Dokumente ab und
generiert einen formellen Prüfbericht – so wie es ein BaFin- oder AMLA-Prüfer tut.

---

## ✨ Features

- 🔍 **RAG-basiertes Retrieval** – Jede Prüffrage holt sich präzise die relevanten Dokumentenstellen
- 📄 **Multi-Modal Ingestion** – PDF, Excel, Interview-Fragebögen, Screenshots, Systemlogs
- 🧑‍⚖️ **BaFin-kalibrierter Prüfer-Agent** – System-Prompt nach echten Prüfungsstandards
- 📊 **Formeller Prüfbericht** – JSON + Markdown + druckfähiges HTML mit Mängelkatalog
- 🔌 **Erweiterbar** – Eigene Kataloge für jede Regulatorik einsteckbar
- ⚡ **Teilprüfungen** – Einzelne Sektionen isoliert prüfen

---

## 🗺️ Unterstützte Regulatorik

| Regulatorik | Status | Prüffelder | Rechtsgrundlage |
|---|---|---|---|
| **GwG / AML** | ✅ Verfügbar | 34 | GwG, §25h KWG, BaFin AuA |
| **MaRisk** | ✅ Verfügbar | 22 | MaRisk AT/BT, §25a KWG |
| **DORA** | ✅ Verfügbar | 18 | DORA Art. 5-46, RTS |
| **WpHG / MaComp** | ✅ Verfügbar | 20 | WpHG, MaComp, MAR, MiFID II |

---

## 🏗️ Architektur

```
finreg-agents/
│
├── pipeline.py              ← Hauptorchestrator (CLI + Python API)
│
├── catalog/
│   └── gwg_catalog.json     ← GwG-Prüfkatalog (34 Prüffelder, 8 Sektionen)
│
├── ingestion/
│   └── ingestor.py          ← Multi-Modal Document Ingestor
│   └── interviews/          ← Beispiel-Fragebögen
│
├── agents/
│   └── pruef_agent.py       ← RAG + LLM Prüfer-Agent
│
└── reports/
    └── bericht_generator.py ← Prüfbericht (JSON / MD / HTML)
```

### Datenfluss

```
Dokumente (PDF, Excel, Interview, Screenshot, Log)
        │
        ▼
  [GwGIngestor]          Multi-Modal Ingestion & Chunking
        │
        ▼
  [VectorStoreIndex]     LlamaIndex + OpenAI Embeddings
        │
        ▼
  [Prüfkatalog]          34 Prüffelder in 8 Sektionen
        │
        │   für jedes Prüffeld:
        ▼
  [GwGPrueferAgent]      RAG-Retrieval → Claude-Bewertung → Befund
        │
        ▼
  [BerichtGenerator]     Mängelkatalog + Prüfbericht (JSON / MD / HTML)
```

---

## 🚀 Quickstart

### 1. Installation

```bash
git clone https://github.com/deinname/finreg-agents.git
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
  interviews/     → Befragungsbögen (*.json)
  screenshots/    → TM-System, goAML, KYC-Oberfläche (*.png, *.jpg)
  logs/           → Systemlogs, Auditlogs (*.txt, *.log)
```

### 4. Prüfung starten

```bash
# GwG-Sonderprüfung (AML)
python pipeline.py --input ./docs --institution "Musterbank AG" --regulatorik gwg

# Nur bestimmte Sektionen (Schnellprüfung)
python pipeline.py --input ./docs --sektionen S01 S02 S05

# Ergebnis-Ordner festlegen
python pipeline.py --input ./docs --institution "Bank XY" --output ./ergebnisse
```

### 5. Python API

```python
from pipeline import GwGAuditPipeline

pipeline = GwGAuditPipeline(
    input_dir="./meine_dokumente",
    institution="Musterbank AG",
    sektionen_filter=["S01", "S02", "S03"],  # optional: Teilprüfung
)
report_paths = pipeline.run()
# → {"json": "./reports/output/gwg_pruefbericht_20250201.json",
#    "markdown": "...", "html": "..."}
```

---

## 📋 GwG-Prüfkatalog

Angelehnt an den realen BaFin-Prüfungsprozess gemäß §44 KWG:

| Sektion | Prüffelder | Rechtsgrundlagen |
|---|---|---|
| S01 · Risikoanalyse | 4 | §5 GwG, §25h Abs.1 KWG |
| S02 · Kundensorgfaltspflichten (KYC) | 6 | §§10–13 GwG |
| S03 · Transaktionsmonitoring | 4 | §25h Abs.2 KWG, §10 Abs.1 Nr.5 GwG |
| S04 · Geldwäschebeauftragter | 3 | §7 GwG, §25h Abs.7 KWG |
| S05 · Verdachtsmeldewesen | 3 | §§43–44 GwG |
| S06 · Schulung & Awareness | 2 | §6 Abs.2 Nr.6 GwG |
| S07 · Aufzeichnungspflichten | 2 | §8 GwG |
| S08 · Interne Revision & Governance | 2 | §25h Abs.5 KWG, MaRisk BT3.2 |

---

## 📊 Bewertungsskala

| Bewertung | Bedeutung |
|---|---|
| ✅ **konform** | Anforderung vollständig erfüllt, Evidenz vorhanden |
| ⚠️ **teilkonform** | Anforderung teilweise erfüllt, Nachbesserung erforderlich |
| 🔴 **nicht_konform** | Anforderung nicht erfüllt – Mangel im Bericht |
| ❓ **nicht_prüfbar** | Keine ausreichende Evidenz im Prüfungskorpus |

**Schweregrade:** `wesentlich` (sofortiger Handlungsbedarf) · `bedeutsam` · `gering`

---

## 🔧 Eigenen Katalog erstellen

Jedes Prüffeld folgt diesem Schema – einfach in eine neue JSON-Datei schreiben
und per `--catalog` übergeben:

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
python pipeline.py --input ./docs --catalog ./catalog/marisk_catalog.json
```

---

## 🗂️ Interview-Fragebogen Format

Strukturierte Befragungsprotokolle werden direkt in den Index aufgenommen:

```json
{
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

---

## 🔮 Roadmap

- [ ] MaRisk-Katalog (AT + BT Module)
- [ ] DORA-Katalog (ICT Risk, Incident Reporting)
- [ ] Persistenter Vektorindex via ChromaDB / Weaviate
- [ ] Claude Vision für Screenshot-Analyse (TM-Systeme, KYC-Oberflächen)
- [ ] Delta-Prüfung – nur geänderte Dokumente neu einlesen
- [ ] Streamlit-UI für interaktive Prüfung
- [ ] Multi-Institut-Vergleich – Benchmarking über Institutsgrenzen

---

## ⚠️ Disclaimer

FinRegAgents ist ein **Simulations- und Vorbereitungstool**. Es ersetzt **keine
offizielle BaFin-Prüfung** und begründet keine Rechtsberatung. Prüfungsergebnisse
sind als interne Vorbereitung zu verstehen, nicht als behördliche Feststellung.

---

## 🤝 Contributing

Contributions willkommen – insbesondere:
- Neue Prüfkataloge (MaRisk, DORA, WpHG)
- Verbesserte Prüffragen und Bewertungskriterien
- Neue Ingestion-Adapter für weitere Dokumenttypen

Bitte fork → branch → PR mit Beschreibung welche Regulatorik erweitert wurde.

---

## 📄 Lizenz

Apache License 2.0 – siehe [LICENSE](LICENSE).

Du kannst FinRegAgents frei nutzen, modifizieren und in kommerzielle Produkte
integrieren, solange der Copyright-Vermerk erhalten bleibt.

---

<div align="center">
  <sub>Gebaut mit LlamaIndex · LangChain · Claude · ❤️</sub>
</div>

---

## 🗂️ Alle Prüfkataloge im Überblick

| Regulatorik | Sektionen | Prüffelder | Schwerpunkte |
|---|---|---|---|
| **GwG** | 8 | 34 | Risikoanalyse, KYC, TM, GwB, SAR, Schulung |
| **DORA** | 5 | 18 | IKT-Risiko, Incident Reporting, TLPT, Drittparteien |
| **MaRisk** | 8 | 22 | Strategie, IKS, RTF, Kredit, Handel, IR, Compliance |
| **WpHG** | 7 | 20 | Compliance, Interessenkonflikte, Geeignetheit, MAR, Best Execution |

```bash
# GwG Sonderprüfung
python pipeline.py --input ./docs --regulatorik gwg

# DORA Prüfung (nur Drittparteienrisiko)
python pipeline.py --input ./docs --regulatorik dora --sektionen D04

# MaRisk Vollprüfung
python pipeline.py --input ./docs --regulatorik marisk --institution "Musterbank AG"

# WpHG / MaComp
python pipeline.py --input ./docs --regulatorik wphg --sektionen W02 W03 W04
```
