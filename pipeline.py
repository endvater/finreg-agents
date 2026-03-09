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
import logging
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings

load_dotenv(override=True)
logger = logging.getLogger(__name__)

from ingestion.ingestor import GwGIngestor
from agents.pruef_agent import PrueferAgent, Sektionsergebnis, SEKTION_REVIEW_ESCALATION
from agents.skeptiker_agent import SkeptikerAgent, merge_befund_skeptiker
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
        local_embeddings: bool = False,
        sektionen_filter: list = None,
        top_k: int = 8,
        verbose: bool = True,
        skeptiker: bool = False,
        skeptiker_only_konform: bool = False,
        adversarial: bool = False,
    ):
        self.input_dir = input_dir
        self.institution = institution
        self.regulatorik = regulatorik
        self.output_dir = output_dir
        self.model = model
        self.embedding_model = embedding_model
        # Lokale Embeddings: explizit gesetzt oder Auto-Fallback wenn kein OPENAI_API_KEY
        self.local_embeddings = local_embeddings or not os.environ.get("OPENAI_API_KEY")
        self.sektionen_filter = sektionen_filter
        self.top_k = top_k
        self.verbose = verbose
        self.skeptiker = skeptiker
        self.skeptiker_only_konform = skeptiker_only_konform
        self.adversarial = adversarial

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
        if self.local_embeddings:
            from llama_index.embeddings.fastembed import FastEmbedEmbedding
            embed_model = FastEmbedEmbedding(model_name="BAAI/bge-small-en-v1.5")
            self._log("   → Embedding: FastEmbed BAAI/bge-small-en-v1.5 (lokal)")
        else:
            from llama_index.embeddings.openai import OpenAIEmbedding
            embed_model = OpenAIEmbedding(model=self.embedding_model)
            self._log(f"   → Embedding: OpenAI {self.embedding_model}")
        # Settings temporär setzen und danach wiederherstellen (save/restore).
        # Beim ersten Aufruf ohne konfigurierten Key schlägt der Getter fehl →
        # _prev_embed auf None setzen und Settings nach dem Lauf zurücksetzen.
        try:
            _prev_embed = Settings.embed_model
        except Exception:
            _prev_embed = None
        Settings.embed_model = embed_model
        try:
            index = VectorStoreIndex.from_documents(documents, show_progress=self.verbose)
        finally:
            if _prev_embed is not None:
                Settings.embed_model = _prev_embed
            else:
                Settings._embed_model = None  # zurück auf "nicht konfiguriert"
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
            adversarial=self.adversarial,
        )
        if self.adversarial:
            self._log("   → Adversarial Prompting Layer aktiviert ⚔️")

        # Skeptiker-Agent optional initialisieren
        skeptiker_agent = None
        if self.skeptiker:
            self._log("   → Skeptiker-Agent aktiviert ⚔️")
            skeptiker_agent = SkeptikerAgent(
                model=self.model,
                only_konform=self.skeptiker_only_konform,
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

                # Skeptiker-Review (optional)
                if skeptiker_agent:
                    t_sk = time.time()
                    skeptiker_result = skeptiker_agent.reviewe(befund, feld)
                    dauer_sk = time.time() - t_sk
                    if not skeptiker_result.akzeptiert:
                        empf = skeptiker_result.bewertung_empfehlung
                        self._log(
                            f"          ⚔️  Skeptiker widerspricht!"
                            f" Empfehlung: {empf.value.upper() if empf else '?'}"
                            f" ({len(skeptiker_result.einwaende)} Einwände, {dauer_sk:.1f}s)"
                        )
                    elif skeptiker_result.einwaende:
                        self._log(
                            f"          ⚔️  Skeptiker: akzeptiert, aber"
                            f" {len(skeptiker_result.einwaende)} Hinweis(e) ({dauer_sk:.1f}s)"
                        )
                    befund = merge_befund_skeptiker(befund, skeptiker_result)

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
        except Exception as e:
            logger.warning("Checkpoint-Fehler (Pipeline läuft weiter): %s", e)

    def _log(self, msg: str):
        if self.verbose:
            print(msg)


# Rückwärtskompatibilität
GwGAuditPipeline = AuditPipeline


# ------------------------------------------------------------------ #
# CLI
# ------------------------------------------------------------------ #
def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )
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
  python pipeline.py --input ./docs --regulatorik gwg --adversarial
  python pipeline.py --input ./docs --regulatorik gwg --adversarial --skeptiker
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
    parser.add_argument("--skeptiker",    action="store_true", default=False,
                        help="Skeptiker-Agent aktivieren (adversariales Review)")
    parser.add_argument("--skeptiker-only-konform", action="store_true", default=False,
                        help="Skeptiker nur für 'konform'-Ratings aktivieren")
    parser.add_argument("--adversarial", action="store_true", default=False,
                        help="Adversarial Prompting Layer: zweiter LLM-Pass mit umgekehrtem "
                             "System-Prompt auf gleicher Evidenz. Große Abweichung → "
                             "Confidence-Penalty + Review-Markierung.")
    parser.add_argument("--local-embeddings", action="store_true", default=False,
                        help="Lokale Embeddings (FastEmbed, kein OpenAI-Key nötig). "
                             "Wird automatisch aktiviert wenn OPENAI_API_KEY fehlt.")
    args = parser.parse_args()

    pipeline = AuditPipeline(
        input_dir=args.input,
        institution=args.institution,
        regulatorik=args.regulatorik,
        catalog_path=args.catalog,
        output_dir=args.output,
        model=args.model,
        local_embeddings=args.local_embeddings,
        sektionen_filter=args.sektionen,
        top_k=args.top_k,
        skeptiker=args.skeptiker,
        skeptiker_only_konform=args.skeptiker_only_konform,
        adversarial=args.adversarial,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
