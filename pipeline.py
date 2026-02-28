"""
GwG Audit Pipeline – Hauptorchestrator
Führt die gesamte Prüfung durch: Ingest → Index → Prüfen → Bericht

Verwendung:
    python pipeline.py --input ./meine_dokumente --institution "Musterbank AG"

Oder als Python-Modul:
    from pipeline import GwGAuditPipeline
    pipeline = GwGAuditPipeline(input_dir="./docs", institution="Musterbank AG")
    report_paths = pipeline.run()
"""

import argparse
import json
import time
from pathlib import Path
from datetime import datetime

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.anthropic import Anthropic as AnthropicLLM

from ingestion.ingestor import GwGIngestor
from agents.pruef_agent import GwGPrueferAgent, SektionsergebniS
from reports.bericht_generator import GwGBerichtGenerator


class GwGAuditPipeline:
    """
    Vollständige GwG-Sonderprüfungs-Pipeline.

    Pipeline-Schritte:
    1. Dokumenten-Ingestion (PDF, Excel, Interview, Screenshot, Log)
    2. Vektorindex aufbauen (LlamaIndex)
    3. Prüfkatalog laden
    4. Für jedes Prüffeld: RAG → LLM-Bewertung → Befund
    5. Prüfbericht generieren (JSON + MD + HTML)
    """

    def __init__(
        self,
        input_dir: str,
        institution: str = "Prüfinstitut",
        catalog_path: str = None,
        output_dir: str = "./reports/output",
        model: str = "claude-opus-4-5",
        embedding_model: str = "text-embedding-3-small",
        sektionen_filter: list[str] = None,
        top_k: int = 8,
        verbose: bool = True,
    ):
        self.input_dir = input_dir
        self.institution = institution
        self.output_dir = output_dir
        self.model = model
        self.sektionen_filter = sektionen_filter  # z.B. ["S01", "S02"] für Teilprüfung
        self.top_k = top_k
        self.verbose = verbose

        # Katalogpfad
        if catalog_path is None:
            catalog_path = Path(__file__).parent / "catalog" / "gwg_catalog.json"
        self.catalog_path = catalog_path

        # LlamaIndex-Einstellungen
        Settings.embed_model = OpenAIEmbedding(model=embedding_model)
        Settings.llm = AnthropicLLM(model=model)

    def run(self) -> dict[str, str]:
        """Führt die komplette Pipeline aus. Gibt Pfade zu den Berichten zurück."""
        t_start = time.time()
        self._log("🚀 GwG-Sonderprüfungs-Pipeline gestartet")
        self._log(f"   Institut: {self.institution}")
        self._log(f"   Eingabeverzeichnis: {self.input_dir}")
        self._log("")

        # ── Schritt 1: Ingestion ─────────────────────────────────────────
        self._log("📂 Schritt 1/4: Dokumenten-Ingestion")
        ingestor = GwGIngestor()
        documents = ingestor.ingest_directory(self.input_dir)
        self._log(f"   → {len(documents)} Dokument-Chunks geladen")

        if not documents:
            raise ValueError(
                f"Keine Dokumente in '{self.input_dir}' gefunden. "
                "Bitte Unterordner pdfs/, excel/, interviews/, screenshots/, logs/ prüfen."
            )

        # ── Schritt 2: Vektorindex ───────────────────────────────────────
        self._log("\n🔍 Schritt 2/4: Vektorindex aufbauen")
        index = VectorStoreIndex.from_documents(documents, show_progress=self.verbose)
        self._log("   → Index fertig")

        # ── Schritt 3: Prüfkatalog laden ─────────────────────────────────
        self._log("\n📋 Schritt 3/4: Prüfkatalog laden & Prüfung durchführen")
        katalog = json.loads(Path(self.catalog_path).read_text(encoding="utf-8"))
        agent = GwGPrueferAgent(index=index, model=self.model, top_k=self.top_k)

        sektionsergebnisse = []
        total_felder = 0
        gepruefte_felder = 0

        for sektion in katalog["pruefsektionen"]:
            # Optionaler Filter für Teilprüfungen
            if self.sektionen_filter and sektion["id"] not in self.sektionen_filter:
                continue

            self._log(f"\n  📌 {sektion['id']}: {sektion['titel']}")
            ergebnis = SektionsergebniS(sektion_id=sektion["id"], titel=sektion["titel"])

            for prueffeld in sektion["prueffelder"]:
                # Rechtsgrundlagen in Prueffeld einbetten (für den Agent)
                prueffeld["rechtsgrundlagen"] = sektion.get("rechtsgrundlagen", [])
                total_felder += 1

                self._log(f"    [{prueffeld['id']}] {prueffeld['frage'][:80]}...")
                t0 = time.time()
                befund = agent.pruefe_feld(prueffeld)
                dauer = time.time() - t0

                status_icon = {
                    "konform": "✅", "teilkonform": "⚠️",
                    "nicht_konform": "🔴", "nicht_prüfbar": "❓"
                }.get(befund.bewertung.value, "?")

                self._log(f"       → {status_icon} {befund.bewertung.value.upper()} ({dauer:.1f}s)")
                ergebnis.befunde.append(befund)
                gepruefte_felder += 1

            sektionsergebnisse.append(ergebnis)

        # ── Schritt 4: Berichte generieren ───────────────────────────────
        self._log(f"\n📝 Schritt 4/4: Prüfberichte generieren")
        generator = GwGBerichtGenerator(
            institution=self.institution,
            pruefer="GwG KI-Prüfungssystem v1.0"
        )
        report_paths = generator.generiere_alle_berichte(
            sektionsergebnisse=sektionsergebnisse,
            output_dir=self.output_dir
        )

        # ── Zusammenfassung ──────────────────────────────────────────────
        t_total = time.time() - t_start
        self._log(f"\n{'='*60}")
        self._log(f"✅ Prüfung abgeschlossen in {t_total:.0f}s")
        self._log(f"   Prüffelder gesamt: {gepruefte_felder}/{total_felder}")
        self._log(f"   Berichte:")
        for fmt, pth in report_paths.items():
            self._log(f"     {fmt.upper()}: {pth}")

        return report_paths

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# ------------------------------------------------------------------ #
# CLI-Einstiegspunkt
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="GwG-Sonderprüfungs-Pipeline (KI-gestützt)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  # Vollständige Prüfung
  python pipeline.py --input ./meine_dokumente --institution "Musterbank AG"

  # Nur bestimmte Sektionen prüfen
  python pipeline.py --input ./docs --sektionen S01 S02 S03

  # Mit eigenem Katalog
  python pipeline.py --input ./docs --catalog ./mein_katalog.json
        """
    )
    parser.add_argument("--input", required=True, help="Verzeichnis mit Prüfungsdokumenten")
    parser.add_argument("--institution", default="Prüfinstitut", help="Name des Instituts")
    parser.add_argument("--output", default="./reports/output", help="Ausgabeverzeichnis für Berichte")
    parser.add_argument("--catalog", default=None, help="Pfad zum GwG-Katalog JSON")
    parser.add_argument("--model", default="claude-opus-4-5", help="Anthropic-Modell")
    parser.add_argument("--sektionen", nargs="*", help="Nur diese Sektionen prüfen (z.B. S01 S02)")
    parser.add_argument("--top-k", type=int, default=8, help="Anzahl RAG-Chunks pro Prüffrage")
    args = parser.parse_args()

    pipeline = GwGAuditPipeline(
        input_dir=args.input,
        institution=args.institution,
        catalog_path=args.catalog,
        output_dir=args.output,
        model=args.model,
        sektionen_filter=args.sektionen,
        top_k=args.top_k,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
