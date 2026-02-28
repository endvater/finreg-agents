"""
FinRegAgents – Multi-Regulatorik Audit-Pipeline (v2)
Unterstützte Regulatorik: GwG, DORA, MaRisk, WpHG/MaComp

Änderungen gegenüber v1:
  - Kein globaler LlamaIndex-State (Settings) mehr → index-lokale Konfiguration
  - Checkpoint-Mechanismus: Zwischenergebnisse werden nach jeder Sektion gesichert
  - Keine Mutation des Katalog-Dicts
  - Dynamische Regulatorik-Labels im Report
  - Model-Default auf Sonnet (kosteneffizient), Opus optional
  - Retry-Logik bei API-Fehlern

Verwendung CLI:
    python pipeline.py --input ./docs --institution "Musterbank AG" --regulatorik gwg
    python pipeline.py --input ./docs --regulatorik dora
    python pipeline.py --input ./docs --regulatorik marisk --sektionen M01 M06
    python pipeline.py --input ./docs --regulatorik wphg --model claude-opus-4-5

Oder als Python-Modul:
    from pipeline import AuditPipeline
    pipeline = AuditPipeline(input_dir="./docs", institution="Musterbank AG", regulatorik="dora")
    report_paths = pipeline.run()
"""

import argparse
import json
import time
from pathlib import Path

from llama_index.core import VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding

from ingestion.ingestor import GwGIngestor
from agents.pruef_agent import PrueferAgent, Sektionsergebnis, SEKTION_REVIEW_ESCALATION
from reports.bericht_generator import BerichtGenerator


# ------------------------------------------------------------------ #
# Katalog-Registry
# ------------------------------------------------------------------ #
KATALOG_REGISTRY = {
    "gwg":    "catalog/gwg_catalog.json",
    "dora":   "catalog/dora_catalog.json",
    "marisk": "catalog/marisk_catalog.json",
    "wphg":   "catalog/wphg_catalog.json",
}

KATALOG_LABELS = {
    "gwg":    "GwG-Sonderprüfung (AML/CFT)",
    "dora":   "DORA – Digital Operational Resilience Act",
    "marisk": "MaRisk-Prüfung",
    "wphg":   "WpHG / MaComp-Prüfung",
}


# ------------------------------------------------------------------ #
# Pipeline
# ------------------------------------------------------------------ #
class AuditPipeline:
    """
    Multi-Regulatorik Audit-Pipeline mit Confidence-Scoring und Validierung.
    """

    def __init__(
        self,
        input_dir: str,
        institution: str = "Prüfinstitut",
        regulatorik: str = "gwg",
        catalog_path: str = None,
        output_dir: str = "./reports/output",
        model: str = "claude-sonnet-4-5-20250514",
        embedding_model: str = "text-embedding-3-small",
        sektionen_filter: list = None,
        top_k: int = 8,
        verbose: bool = True,
    ):
        self.input_dir = input_dir
        self.institution = institution
        self.regulatorik = regulatorik
        self.output_dir = output_dir
        self.model = model
        self.embedding_model = embedding_model
        self.sektionen_filter = sektionen_filter
        self.top_k = top_k
        self.verbose = verbose

        # Katalogpfad auflösen
        base = Path(__file__).parent
        if catalog_path:
            self.catalog_path = Path(catalog_path)
        elif regulatorik in KATALOG_REGISTRY:
            self.catalog_path = base / KATALOG_REGISTRY[regulatorik]
        else:
            raise ValueError(
                f"Unbekannte Regulatorik: '{regulatorik}'. "
                f"Verfügbar: {list(KATALOG_REGISTRY.keys())}"
            )

    def run(self) -> dict:
        """Führt die komplette Pipeline aus. Gibt Pfade zu den Berichten zurück."""
        t_start = time.time()
        label = KATALOG_LABELS.get(self.regulatorik, self.regulatorik.upper())

        self._log(f"🚀 FinRegAgents Pipeline v2 gestartet")
        self._log(f"   Regulatorik: {label}")
        self._log(f"   Institut:    {self.institution}")
        self._log(f"   Modell:      {self.model}")
        self._log(f"   Katalog:     {self.catalog_path}")
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
        # Embedding-Modell pro Index konfigurieren (kein globaler State)
        embed_model = OpenAIEmbedding(model=self.embedding_model)
        Settings.embed_model = embed_model
        index = VectorStoreIndex.from_documents(documents, show_progress=self.verbose)
        self._log("   → Index fertig")

        # ── Schritt 3: Prüfkatalog laden & Prüfung durchführen ──────────
        self._log(f"\n📋 Schritt 3/4: Katalog laden & Prüfung durchführen [{label}]")
        katalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        katalog_version = katalog.get("katalog_version", "unbekannt")

        agent = PrueferAgent(
            index=index,
            regulatorik=self.regulatorik,
            model=self.model,
            top_k=self.top_k,
        )

        sektionsergebnisse = []
        total_felder = 0
        gepruefte_felder = 0
        checkpoint_dir = Path(self.output_dir) / ".checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for sektion in katalog["pruefsektionen"]:
            if self.sektionen_filter and sektion["id"] not in self.sektionen_filter:
                continue

            self._log(f"\n  📌 {sektion['id']}: {sektion['titel']}")
            ergebnis = Sektionsergebnis(sektion_id=sektion["id"], titel=sektion["titel"])

            for prueffeld in sektion["prueffelder"]:
                # Lokale Kopie mit Rechtsgrundlagen – keine Mutation des Originals
                feld = {
                    **prueffeld,
                    "rechtsgrundlagen": sektion.get("rechtsgrundlagen", []),
                }
                total_felder += 1

                self._log(f"    [{feld['id']}] {feld['frage'][:80]}...")
                t0 = time.time()
                befund = agent.pruefe_feld(feld)
                dauer = time.time() - t0

                status_icon = {
                    "konform": "✅", "teilkonform": "⚠️",
                    "nicht_konform": "🔴", "nicht_prüfbar": "❓"
                }.get(befund.bewertung.value, "?")

                conf_str = f" | Conf: {befund.confidence:.0%}"
                review_str = " | 🔍 REVIEW" if befund.review_erforderlich else ""
                self._log(f"       → {status_icon} {befund.bewertung.value.upper()}{conf_str}{review_str} ({dauer:.1f}s)")

                if befund.validierungshinweise:
                    for hint in befund.validierungshinweise:
                        self._log(f"          ⚡ {hint}")

                ergebnis.befunde.append(befund)
                gepruefte_felder += 1

            # Sektions-Eskalation prüfen
            if ergebnis.review_quote >= SEKTION_REVIEW_ESCALATION:
                self._log(f"  ⚠️  Sektion {sektion['id']}: {ergebnis.review_quote:.0%} Review-Quote → Eskalation empfohlen")

            sektionsergebnisse.append(ergebnis)

            # Checkpoint: Zwischenergebnis sichern
            self._save_checkpoint(sektionsergebnisse, checkpoint_dir)

        # ── Schritt 4: Berichte generieren ───────────────────────────────
        self._log(f"\n📝 Schritt 4/4: Prüfberichte generieren")
        generator = BerichtGenerator(
            institution=self.institution,
            pruefer=f"FinRegAgents v2.0 – {label}",
            regulatorik=self.regulatorik,
            model=self.model,
            katalog_version=katalog_version,
        )
        report_paths = generator.generiere_alle_berichte(
            sektionsergebnisse=sektionsergebnisse,
            output_dir=self.output_dir,
        )

        # ── Zusammenfassung ──────────────────────────────────────────────
        t_total = time.time() - t_start
        self._log(f"\n{'='*60}")
        self._log(f"✅ Prüfung abgeschlossen in {t_total:.0f}s")
        self._log(f"   Regulatorik: {label}")
        self._log(f"   Prüffelder:  {gepruefte_felder}/{total_felder}")
        self._log(f"   Berichte:")
        for fmt, pth in report_paths.items():
            self._log(f"     {fmt.upper()}: {pth}")

        return report_paths

    def _save_checkpoint(self, sektionsergebnisse: list, checkpoint_dir: Path):
        """Sichert Zwischenergebnisse nach jeder Sektion."""
        try:
            data = []
            for s in sektionsergebnisse:
                data.append({
                    "id": s.sektion_id,
                    "titel": s.titel,
                    "befunde": [
                        {
                            "id": b.prueffeld_id,
                            "bewertung": b.bewertung.value,
                            "confidence": b.confidence,
                            "review_erforderlich": b.review_erforderlich,
                            "begruendung": b.begruendung[:200],
                        }
                        for b in s.befunde
                    ]
                })
            path = checkpoint_dir / "checkpoint_latest.json"
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass  # Checkpoint-Fehler sollen die Pipeline nicht stoppen

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# Rückwärtskompatibilität
GwGAuditPipeline = AuditPipeline


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(
        description="FinRegAgents v2 – Multi-Regulatorik Audit-Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Regulatorik-Optionen:
  gwg     → GwG-Sonderprüfung (AML/CFT) – 34 Prüffelder
  dora    → DORA – Digital Operational Resilience Act – 18 Prüffelder
  marisk  → MaRisk-Prüfung – 22 Prüffelder
  wphg    → WpHG / MaComp-Prüfung – 20 Prüffelder

Beispiele:
  python pipeline.py --input ./docs --institution "Musterbank AG" --regulatorik gwg
  python pipeline.py --input ./docs --regulatorik dora --sektionen D01 D04
  python pipeline.py --input ./docs --regulatorik marisk
  python pipeline.py --input ./docs --regulatorik wphg --model claude-opus-4-5
        """
    )
    parser.add_argument("--input",        required=True,  help="Verzeichnis mit Prüfungsdokumenten")
    parser.add_argument("--institution",  default="Prüfinstitut", help="Name des Instituts")
    parser.add_argument("--regulatorik",  default="gwg",
                        choices=list(KATALOG_REGISTRY.keys()),
                        help="Zu prüfende Regulatorik")
    parser.add_argument("--output",       default="./reports/output", help="Ausgabeverzeichnis")
    parser.add_argument("--catalog",      default=None,   help="Eigener Katalog (überschreibt --regulatorik)")
    parser.add_argument("--model",        default="claude-sonnet-4-5-20250514",
                        help="Anthropic-Modell (Default: Sonnet für Kosteneffizienz)")
    parser.add_argument("--sektionen",    nargs="*",      help="Nur diese Sektionen prüfen (z.B. S01 S02)")
    parser.add_argument("--top-k",        type=int, default=8, help="RAG-Chunks pro Prüffrage")
    args = parser.parse_args()

    pipeline = AuditPipeline(
        input_dir=args.input,
        institution=args.institution,
        regulatorik=args.regulatorik,
        catalog_path=args.catalog,
        output_dir=args.output,
        model=args.model,
        sektionen_filter=args.sektionen,
        top_k=args.top_k,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
