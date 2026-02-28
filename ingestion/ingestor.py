"""
FinRegAgents – Multi-Modal Document Ingestor (v2)
Unterstützt: PDF, Excel/CSV, Interview-Fragebögen (JSON/YAML), Screenshots, Logs

Änderungen gegenüber v1:
  - Chunk-Size auf 1024 erhöht (besser für regulatorische Dokumente)
  - Interview-JSON: unterstützt sowohl Array- als auch Dict-Format (fragen_antworten)
  - YAML-Support mit yaml.safe_load()
  - Screenshots: base64 wird NICHT mehr in Index-Metadaten gespeichert (Memory-Fix)
  - Excel: Aggregations-Summary statt nur erste 50 Zeilen
  - Deduplizierung über Datei-Hash
"""

import json
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


class GwGIngestor:
    """
    Lädt alle Eingabedokumente und wandelt sie in LlamaIndex-Documents um,
    angereichert mit Metadaten (input_type, source, file_hash).
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 128):
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.pdf_reader = PDFReader()
        self._seen_hashes: set[str] = set()

    # ------------------------------------------------------------------ #
    # Public: Haupteinstiegspunkt
    # ------------------------------------------------------------------ #
    def ingest_directory(self, base_path: str) -> list[Document]:
        """
        Scannt ein Verzeichnis und ingestiert alle unterstützten Dateien.
        Erwartete Struktur:
            base_path/
              pdfs/           → Policies, Prozessbeschreibungen
              excel/          → Datenbankabzüge, Auswertungen
              interviews/     → JSON/YAML-Fragebögen
              screenshots/    → System-Screenshots (PNG/JPG)
              logs/           → Systemlogs (TXT/CSV)
        """
        base = Path(base_path)
        all_docs: list[Document] = []

        folder_handlers = {
            "pdfs":        self._ingest_pdfs,
            "excel":       self._ingest_excel,
            "interviews":  self._ingest_interviews,
            "screenshots": self._ingest_screenshots,
            "logs":        self._ingest_logs,
        }

        for folder_name, handler in folder_handlers.items():
            folder = base / folder_name
            if folder.exists():
                print(f"  📂 Ingesting {folder_name}/ ...")
                docs = handler(folder)
                all_docs.extend(docs)
                print(f"     → {len(docs)} Dokument(e) / Chunks geladen")

        print(f"  📊 Gesamt: {len(all_docs)} Chunks | Deduplizierung: {len(self._seen_hashes)} unique Dateien")
        return all_docs

    # ------------------------------------------------------------------ #
    # Deduplizierung
    # ------------------------------------------------------------------ #
    def _file_hash(self, path: Path) -> str:
        """SHA-256 Hash einer Datei für Deduplizierung."""
        h = hashlib.sha256()
        h.update(path.read_bytes())
        return h.hexdigest()

    def _is_duplicate(self, path: Path) -> bool:
        """Prüft ob eine Datei bereits ingested wurde (identischer Inhalt)."""
        fh = self._file_hash(path)
        if fh in self._seen_hashes:
            print(f"     ⏭ Duplikat übersprungen: {path.name}")
            return True
        self._seen_hashes.add(fh)
        return False

    # ------------------------------------------------------------------ #
    # PDF-Ingestion
    # ------------------------------------------------------------------ #
    def _ingest_pdfs(self, folder: Path) -> list[Document]:
        docs = []
        for pdf_file in sorted(folder.glob("*.pdf")):
            if self._is_duplicate(pdf_file):
                continue
            try:
                raw = self.pdf_reader.load_data(str(pdf_file))
                for doc in raw:
                    doc.metadata.update({
                        "input_type": "pdf",
                        "source": pdf_file.name,
                        "file_path": str(pdf_file),
                    })
                chunks = self.splitter.get_nodes_from_documents(raw)
                for node in chunks:
                    docs.append(Document(
                        text=node.get_content(),
                        metadata=node.metadata,
                    ))
            except Exception as e:
                print(f"     ⚠ Fehler bei {pdf_file.name}: {e}")
        return docs

    # ------------------------------------------------------------------ #
    # Excel / CSV-Ingestion
    # ------------------------------------------------------------------ #
    def _ingest_excel(self, folder: Path) -> list[Document]:
        docs = []
        patterns = ["*.xlsx", "*.xls", "*.csv"]
        files = sorted(f for pat in patterns for f in folder.glob(pat))
        for f in files:
            if self._is_duplicate(f):
                continue
            try:
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)

                text = self._dataframe_to_text(df, f.name)
                docs.append(Document(
                    text=text,
                    metadata={
                        "input_type": "excel",
                        "source": f.name,
                        "file_path": str(f),
                        "rows": len(df),
                        "columns": list(df.columns.astype(str)),
                    }
                ))
            except Exception as e:
                print(f"     ⚠ Fehler bei {f.name}: {e}")
        return docs

    def _dataframe_to_text(self, df: pd.DataFrame, filename: str) -> str:
        """Wandelt DataFrame in einen prüfbaren Textblock um."""
        lines = [f"=== Datei: {filename} ==="]
        lines.append(f"Spalten: {', '.join(df.columns.astype(str))}")
        lines.append(f"Anzahl Zeilen: {len(df)}")
        lines.append("")

        # Aggregations-Statistiken für numerische Spalten
        num_cols = df.select_dtypes(include="number").columns
        if len(num_cols) > 0:
            lines.append("Numerische Zusammenfassung:")
            lines.append(df[num_cols].describe().to_string())
            lines.append("")

        # Kategorische Spalten: Häufigkeitsverteilung
        cat_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in cat_cols[:5]:  # Max 5 Spalten
            vc = df[col].value_counts().head(10)
            if len(vc) > 0:
                lines.append(f"Verteilung '{col}': {dict(vc)}")
        if cat_cols.any():
            lines.append("")

        # Datenauszug: Erste und letzte Zeilen
        head_n = min(30, len(df))
        lines.append(f"Datenauszug (erste {head_n} Zeilen):")
        lines.append(df.head(head_n).to_string(index=False))

        if len(df) > 60:
            lines.append(f"\n... ({len(df) - 60} Zeilen ausgelassen) ...\n")
            lines.append(f"Letzte 30 Zeilen:")
            lines.append(df.tail(30).to_string(index=False))

        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Interview-Fragebögen (JSON / YAML)
    # ------------------------------------------------------------------ #
    def _ingest_interviews(self, folder: Path) -> list[Document]:
        docs = []
        patterns = ["*.json", "*.yaml", "*.yml", "*.txt"]
        files = sorted(f for pat in patterns for f in folder.glob(pat))
        for f in files:
            if self._is_duplicate(f):
                continue
            try:
                if f.suffix == ".json":
                    data = json.loads(f.read_text(encoding="utf-8"))
                    text = self._interview_data_to_text(data, f.name)
                elif f.suffix in (".yaml", ".yml"):
                    if HAS_YAML:
                        data = yaml.safe_load(f.read_text(encoding="utf-8"))
                        text = self._interview_data_to_text(data, f.name)
                    else:
                        print(f"     ⚠ YAML-Support nicht verfügbar (pip install pyyaml)")
                        text = f.read_text(encoding="utf-8")
                else:
                    text = f.read_text(encoding="utf-8")

                docs.append(Document(
                    text=text,
                    metadata={
                        "input_type": "interview",
                        "source": f.name,
                        "file_path": str(f),
                    }
                ))
            except Exception as e:
                print(f"     ⚠ Fehler bei {f.name}: {e}")
        return docs

    def _interview_data_to_text(self, data: Any, filename: str) -> str:
        """
        Unterstützt beide Formate:
        Format A (Dict): {"meta": {...}, "fragen_antworten": [{...}, ...]}
        Format B (Array): [{"frage": "...", "antwort": "..."}, ...]
        """
        lines = [f"=== Interview-Fragebogen: {filename} ===\n"]

        # Format A: Dict mit meta + fragen_antworten
        if isinstance(data, dict):
            meta = data.get("meta", {})
            if meta:
                for k, v in meta.items():
                    lines.append(f"{k}: {v}")
                lines.append("")

            qa_list = data.get("fragen_antworten", [])
            if qa_list:
                lines.extend(self._format_qa_list(qa_list))
            else:
                # Fallback: Einfache Key-Value-Paare
                for k, v in data.items():
                    if k != "meta":
                        lines.append(f"{k}: {v}")

        # Format B: Direkt ein Array
        elif isinstance(data, list):
            lines.extend(self._format_qa_list(data))

        return "\n".join(lines)

    def _format_qa_list(self, qa_list: list) -> list[str]:
        """Formatiert eine Liste von Frage-Antwort-Paaren."""
        lines = []
        for i, qa in enumerate(qa_list, 1):
            if not isinstance(qa, dict):
                continue
            frage_id = qa.get("id", f"F-{i:02d}")
            ref = qa.get("prueffeld_referenz", "")
            ref_str = f" [Ref: {ref}]" if ref else ""

            lines.append(f"Frage {frage_id}{ref_str}: {qa.get('frage', '')}")
            lines.append(f"Antwort: {qa.get('antwort', '')}")
            if qa.get("kommentar"):
                lines.append(f"Kommentar: {qa['kommentar']}")
            if qa.get("datum"):
                lines.append(f"Datum: {qa['datum']}")
            lines.append("")
        return lines

    # ------------------------------------------------------------------ #
    # Screenshot-Ingestion
    # ------------------------------------------------------------------ #
    def _ingest_screenshots(self, folder: Path) -> list[Document]:
        """
        Screenshots werden als Platzhalter-Dokumente indexiert.
        base64-Daten werden NICHT in den Index geladen (Memory-Optimierung).
        Für Vision-Analyse: Separate Pipeline oder manuelles Review.
        """
        docs = []
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.webp"]
        files = sorted(f for pat in patterns for f in folder.glob(pat))
        for f in files:
            if self._is_duplicate(f):
                continue
            try:
                file_size_kb = f.stat().st_size / 1024
                docs.append(Document(
                    text=(
                        f"[SCREENSHOT: {f.name}] "
                        f"Systemscreenshot ({file_size_kb:.0f} KB). "
                        f"Visuelle Prüfung durch einen menschlichen Prüfer erforderlich. "
                        f"Dieser Screenshot kann relevante Evidenz für KYC-Oberflächen, "
                        f"TM-Systeme oder goAML-Zugang enthalten."
                    ),
                    metadata={
                        "input_type": "screenshot",
                        "source": f.name,
                        "file_path": str(f),
                        "file_size_kb": round(file_size_kb, 1),
                    }
                ))
            except Exception as e:
                print(f"     ⚠ Fehler bei {f.name}: {e}")
        return docs

    # ------------------------------------------------------------------ #
    # Log-Ingestion
    # ------------------------------------------------------------------ #
    def _ingest_logs(self, folder: Path) -> list[Document]:
        docs = []
        patterns = ["*.txt", "*.log", "*.csv"]
        files = sorted(f for pat in patterns for f in folder.glob(pat))
        for f in files:
            if self._is_duplicate(f):
                continue
            try:
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                    text = self._dataframe_to_text(df, f.name)
                else:
                    text = f.read_text(encoding="utf-8", errors="replace")

                base_meta = {"input_type": "log", "source": f.name, "file_path": str(f)}
                chunks = self.splitter.get_nodes_from_documents([
                    Document(text=text, metadata=base_meta)
                ])
                for node in chunks:
                    docs.append(Document(text=node.get_content(), metadata=node.metadata))
            except Exception as e:
                print(f"     ⚠ Fehler bei {f.name}: {e}")
        return docs
