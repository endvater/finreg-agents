"""
GwG Audit Pipeline – Multi-Modal Document Ingestor
Unterstützt: PDF, Excel/CSV, Interview-Fragebögen (JSON/YAML), Screenshots (via Vision)
"""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from PIL import Image
import base64


class GwGIngestor:
    """
    Lädt alle Eingabedokumente und wandelt sie in LlamaIndex-Documents um,
    angereichert mit Metadaten (input_type, source, pruef_sektion_hint).
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.pdf_reader = PDFReader()

    # ------------------------------------------------------------------ #
    # Public: Haupteinstiegspunkt
    # ------------------------------------------------------------------ #
    def ingest_directory(self, base_path: str) -> list[Document]:
        """
        Scannt ein Verzeichnis und ingested alle unterstützten Dateien.
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
            "pdfs":       self._ingest_pdfs,
            "excel":      self._ingest_excel,
            "interviews": self._ingest_interviews,
            "screenshots": self._ingest_screenshots,
            "logs":       self._ingest_logs,
        }

        for folder_name, handler in folder_handlers.items():
            folder = base / folder_name
            if folder.exists():
                print(f"  📂 Ingesting {folder_name}/ ...")
                docs = handler(folder)
                all_docs.extend(docs)
                print(f"     → {len(docs)} Dokument(e) / Chunks geladen")

        return all_docs

    # ------------------------------------------------------------------ #
    # PDF-Ingestion
    # ------------------------------------------------------------------ #
    def _ingest_pdfs(self, folder: Path) -> list[Document]:
        docs = []
        for pdf_file in folder.glob("*.pdf"):
            raw = self.pdf_reader.load_data(str(pdf_file))
            for doc in raw:
                doc.metadata.update({
                    "input_type": "pdf",
                    "source": pdf_file.name,
                    "file_path": str(pdf_file),
                })
            chunks = self.splitter.get_nodes_from_documents(raw)
            # Nodes → Documents
            for node in chunks:
                docs.append(Document(
                    text=node.get_content(),
                    metadata=node.metadata,
                ))
        return docs

    # ------------------------------------------------------------------ #
    # Excel / CSV-Ingestion
    # ------------------------------------------------------------------ #
    def _ingest_excel(self, folder: Path) -> list[Document]:
        docs = []
        for f in list(folder.glob("*.xlsx")) + list(folder.glob("*.xls")) + list(folder.glob("*.csv")):
            try:
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                else:
                    df = pd.read_excel(f)

                # Konvertiere jedes Sheet / jede Tabelle in strukturierten Text
                summary = self._dataframe_to_text(df, f.name)
                docs.append(Document(
                    text=summary,
                    metadata={
                        "input_type": "excel",
                        "source": f.name,
                        "file_path": str(f),
                        "rows": len(df),
                        "columns": list(df.columns),
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
        # Erste 50 Zeilen als Textrepräsentation
        lines.append("Datenauszug (erste 50 Zeilen):")
        lines.append(df.head(50).to_string(index=False))
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Interview-Fragebögen (JSON)
    # ------------------------------------------------------------------ #
    def _ingest_interviews(self, folder: Path) -> list[Document]:
        docs = []
        for f in list(folder.glob("*.json")) + list(folder.glob("*.yaml")) + list(folder.glob("*.txt")):
            try:
                if f.suffix == ".json":
                    data = json.loads(f.read_text(encoding="utf-8"))
                    text = self._interview_json_to_text(data, f.name)
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

    def _interview_json_to_text(self, data: Any, filename: str) -> str:
        """
        Erwartet Format:
        [{"frage": "...", "antwort": "...", "datum": "...", "interviewer": "..."}, ...]
        """
        lines = [f"=== Interview-Fragebogen: {filename} ===\n"]
        if isinstance(data, list):
            for i, qa in enumerate(data, 1):
                lines.append(f"Frage {i}: {qa.get('frage', '')}")
                lines.append(f"Antwort: {qa.get('antwort', '')}")
                if qa.get("kommentar"):
                    lines.append(f"Kommentar: {qa['kommentar']}")
                lines.append("")
        elif isinstance(data, dict):
            for k, v in data.items():
                lines.append(f"{k}: {v}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Screenshot-Ingestion (Vision via base64)
    # ------------------------------------------------------------------ #
    def _ingest_screenshots(self, folder: Path) -> list[Document]:
        docs = []
        for f in list(folder.glob("*.png")) + list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")):
            try:
                # Encode als base64 – wird später vom Vision-fähigen LLM interpretiert
                with open(f, "rb") as img_file:
                    b64 = base64.b64encode(img_file.read()).decode("utf-8")

                docs.append(Document(
                    text=f"[SCREENSHOT: {f.name}] Dieses Bild ist als base64 in den Metadaten gespeichert.",
                    metadata={
                        "input_type": "screenshot",
                        "source": f.name,
                        "file_path": str(f),
                        "image_base64": b64,
                        "image_suffix": f.suffix.lstrip("."),
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
        for f in list(folder.glob("*.txt")) + list(folder.glob("*.log")) + list(folder.glob("*.csv")):
            try:
                if f.suffix == ".csv":
                    df = pd.read_csv(f)
                    text = self._dataframe_to_text(df, f.name)
                else:
                    text = f.read_text(encoding="utf-8", errors="replace")

                # Chunk Logs
                chunks = self.splitter.get_nodes_from_documents([
                    Document(text=text, metadata={"input_type": "log", "source": f.name})
                ])
                for node in chunks:
                    docs.append(Document(text=node.get_content(), metadata=node.metadata))
            except Exception as e:
                print(f"     ⚠ Fehler bei {f.name}: {e}")
        return docs
