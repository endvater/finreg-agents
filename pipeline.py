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
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, Settings

from ingestion.ingestor import GwGIngestor
from ingestion.relevance_classifier import EvidenceRelevanceClassifier
from agents.pruef_agent import PrueferAgent, Sektionsergebnis, SEKTION_REVIEW_ESCALATION
from agents.skeptiker_agent import SkeptikerAgent, merge_befund_skeptiker
from agents.llm_factory import list_providers, default_model
from agents.embedding_factory import (
    build_embedding,
    list_embedding_providers,
    default_embedding_model,
)
from reports.bericht_generator import BerichtGenerator

load_dotenv(override=True)
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Custom Exceptions
# ------------------------------------------------------------------ #
class ReviewBudgetExceeded(Exception):
    """Raised when the review budget has been reached during the pipeline run."""

    pass


# ------------------------------------------------------------------ #
# Katalog-Registry
# ------------------------------------------------------------------ #
KATALOG_REGISTRY = {
    "gwg": "catalog/gwg_catalog.json",
    "dora": "catalog/dora_catalog.json",
    "marisk": "catalog/marisk_catalog.json",
    "wphg": "catalog/wphg_catalog.json",
}

KATALOG_LABELS = {
    "gwg": "GwG-Sonderprüfung (AML/CFT)",
    "dora": "DORA – Digital Operational Resilience Act",
    "marisk": "MaRisk-Prüfung",
    "wphg": "WpHG / MaComp-Prüfung",
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
        provider: str = "anthropic",
        model: str | None = None,
        embedding_provider: str | None = None,
        embedding_model: str | None = None,
        sektionen_filter: list = None,
        top_k: int = 8,
        verbose: bool = True,
        verbose_token_details: bool = False,
        evidence_relevance_filter: bool = False,
        skeptiker: bool = False,
        skeptiker_only_konform: bool = False,
        adversarial: bool = False,
        use_relevance_filter: bool = False,
        review_budget: int = 0,
        resume: bool = False,
    ):
        self.input_dir = input_dir
        self.institution = institution
        self.regulatorik = regulatorik
        self.output_dir = output_dir
        self.provider = provider
        self.model = model or default_model(provider)
        self.embedding_provider = embedding_provider  # None → Auto-Detect in factory
        self.embedding_model = embedding_model  # None → Provider-Default in factory
        self.sektionen_filter = sektionen_filter
        self.top_k = top_k
        self.verbose = verbose
        self.verbose_token_details = verbose_token_details
        self.review_budget = review_budget
        self.evidence_relevance_filter = evidence_relevance_filter
        self.skeptiker = skeptiker
        self.skeptiker_only_konform = skeptiker_only_konform
        self.run_token_stats = {
            "version": "1.0",
            "gesamt": {"input": 0, "output": 0, "total": 0},
            "nach_agent": {
                "pruefer": {"input": 0, "output": 0, "total": 0},
                "skeptiker": {"input": 0, "output": 0, "total": 0},
            },
            "details": [],
        }
        self.adversarial = adversarial
        self.use_relevance_filter = use_relevance_filter
        self.resume = resume

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

        self._log("🚀 FinRegAgents Pipeline v2 gestartet")
        self._log(f"   Regulatorik: {label}")
        self._log(f"   Institut:    {self.institution}")
        self._log(f"   Provider:    {self.provider}")
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
        embed_model = build_embedding(
            provider=self.embedding_provider,
            model=self.embedding_model,
        )
        ep = self.embedding_provider or "auto"
        em = self.embedding_model or default_embedding_model(
            self.embedding_provider or "openai"
        )
        self._log(f"   → Embedding: {ep} / {em}")
        # Settings temporär setzen und danach wiederherstellen, damit kein
        # dauerhafter globaler State entsteht (wichtig bei mehreren Pipeline-Instanzen)
        try:
            _prev_embed = Settings.embed_model
        except Exception:
            _prev_embed = None
        Settings.embed_model = embed_model
        try:
            index = VectorStoreIndex.from_documents(
                documents, show_progress=self.verbose
            )
        finally:
            if _prev_embed is not None:
                Settings.embed_model = _prev_embed
            else:
                Settings._embed_model = None
        self._log("   → Index fertig")

        # ── Schritt 2b: Relevanz-Filter (Feature-Flag) ───────────────────
        if self.use_relevance_filter:
            self._log("\n🔎 Relevanz-Filter aktiviert (Feature-Flag)")
            classifier = EvidenceRelevanceClassifier()
            kept_docs, dropped_docs = classifier.filter_chunks(
                documents, self.regulatorik
            )
            n_filtered = len(dropped_docs)
            n_kept = len(kept_docs)
            self._log(
                f"   → {n_filtered} Chunks gefiltert (context_noise), {n_kept} behalten"
            )

            # Write sampling report (up to 20 random decisions)
            sample_size = min(20, len(dropped_docs) + len(kept_docs))
            all_decisions = []
            for chunk in dropped_docs:
                all_decisions.append(
                    {
                        "node_id": chunk.node_id,
                        "category": chunk.category,
                        "drop_reason": chunk.drop_reason,
                        "text_preview": chunk.text[:120],
                        "kept": False,
                    }
                )
            for doc in kept_docs[:sample_size]:
                if hasattr(doc, "text"):
                    text_preview = doc.text[:120]
                elif hasattr(doc, "get_content"):
                    text_preview = doc.get_content()[:120]
                else:
                    text_preview = str(doc)[:120]
                all_decisions.append(
                    {
                        "node_id": getattr(doc, "doc_id", None) or "unknown",
                        "category": "kept",
                        "drop_reason": None,
                        "text_preview": text_preview,
                        "kept": True,
                    }
                )
            sample = random.sample(all_decisions, min(20, len(all_decisions)))
            sampling_report = {
                "regulatorik": self.regulatorik,
                "total_chunks": len(documents),
                "kept": n_kept,
                "filtered": n_filtered,
                "filter_rate_pct": round(
                    100.0 * n_filtered / max(1, len(documents)), 1
                ),
                "sample": sample,
            }
            out_dir = Path(self.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            sampling_path = out_dir / "relevance_sampling.json"
            sampling_path.write_text(
                json.dumps(sampling_report, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self._log(f"   → Sampling-Report: {sampling_path}")

            # Rebuild index with filtered documents if any were dropped
            if n_filtered > 0 and n_kept > 0:
                self._log("   → Index wird mit gefilterten Dokumenten neu aufgebaut")
                try:
                    _prev_embed2 = Settings.embed_model
                except Exception:
                    _prev_embed2 = None
                Settings.embed_model = embed_model
                try:
                    index = VectorStoreIndex.from_documents(
                        kept_docs, show_progress=self.verbose
                    )
                finally:
                    if _prev_embed2 is not None:
                        Settings.embed_model = _prev_embed2
                    else:
                        Settings._embed_model = None

        # ── Schritt 3: Prüfkatalog laden & Prüfung durchführen ──────────
        self._log(f"\n📋 Schritt 3/4: Katalog laden & Prüfung durchführen [{label}]")
        katalog = json.loads(self.catalog_path.read_text(encoding="utf-8"))
        katalog_version = katalog.get("katalog_version", "unbekannt")

        agent = PrueferAgent(
            index=index,
            regulatorik=self.regulatorik,
            provider=self.provider,
            model=self.model,
            top_k=self.top_k,
            adversarial=self.adversarial,
            evidence_relevance_filter=self.evidence_relevance_filter,
        )
        if self.adversarial:
            self._log("   → Adversarial Prompting Layer aktiviert ⚔️")

        # Skeptiker-Agent optional initialisieren
        skeptiker_agent = None
        if self.skeptiker:
            self._log("   → Skeptiker-Agent aktiviert ⚔️")
            skeptiker_agent = SkeptikerAgent(
                provider=self.provider,
                model=self.model,
                only_konform=self.skeptiker_only_konform,
            )

        sektionsergebnisse = []
        total_felder = 0
        gepruefte_felder = 0
        review_markierte_felder = 0
        review_budget_erreicht = False
        checkpoint_dir = Path(self.output_dir) / ".checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Resume: load completed section IDs from checkpoint
        completed_sektion_ids: set[str] = set()
        if self.resume:
            completed_sektion_ids = self._load_completed_sektion_ids(checkpoint_dir)
            if completed_sektion_ids:
                self._log(
                    f"   → Resume-Modus: {len(completed_sektion_ids)} bereits abgeschlossene Sektion(en) werden übersprungen."
                )

        review_counter = 0

        for sektion in katalog["pruefsektionen"]:
            if self.sektionen_filter and sektion["id"] not in self.sektionen_filter:
                continue

            # Resume: skip already-completed sections
            if sektion["id"] in completed_sektion_ids:
                self._log(
                    f"\n  ⏭️  Überspringe bereits abgeschlossene Sektion: {sektion['id']}: {sektion['titel']}"
                )
                continue

            self._log(f"\n  📌 {sektion['id']}: {sektion['titel']}")
            ergebnis = Sektionsergebnis(
                sektion_id=sektion["id"], titel=sektion["titel"]
            )

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
                self._add_token_usage("pruefer", befund.token_usage)

                status_icon = {
                    "konform": "✅",
                    "teilkonform": "⚠️",
                    "nicht_konform": "🔴",
                    "nicht_prüfbar": "❓",
                    "disputed": "⚖️",
                }.get(befund.bewertung.value, "?")

                conf_str = (
                    f" | Conf: {befund.confidence:.0%} ({befund.confidence_level})"
                )
                review_str = " | 🔍 REVIEW" if befund.review_erforderlich else ""
                self._log(
                    f"       → {status_icon} {befund.bewertung.value.upper()}{conf_str}{review_str} ({dauer:.1f}s)"
                )

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
                    self._add_token_usage("skeptiker", skeptiker_result.token_usage)

                self._add_detail_stat(sektion["id"], feld["id"], befund)
                ergebnis.befunde.append(befund)
                gepruefte_felder += 1
                if befund.review_erforderlich:
                    review_markierte_felder += 1
                    if (
                        self.review_budget is not None
                        and review_markierte_felder >= self.review_budget
                    ):
                        review_budget_erreicht = True
                        self._log(
                            f"  ⏸️ Review-Budget erreicht ({review_markierte_felder}/{self.review_budget}). "
                            "Lauf wird nach dieser Sektion pausiert."
                        )
                        break

                # Review-Budget-Tracking
                if befund.review_erforderlich and self.review_budget > 0:
                    review_counter += 1
                    if review_counter >= self.review_budget:
                        # Save partial section results before raising
                        sektionsergebnisse.append(ergebnis)
                        self._save_checkpoint(sektionsergebnisse, checkpoint_dir)
                        checkpoint_path = checkpoint_dir / "checkpoint_latest.json"
                        raise ReviewBudgetExceeded(
                            f"Review-Budget von {self.review_budget} erreicht – Prüfung pausiert.\n"
                            f"   Checkpoint gespeichert: {checkpoint_path}\n"
                            f"   Weiter mit: python pipeline.py ... --resume"
                        )

            # Sektions-Eskalation prüfen
            if ergebnis.review_quote >= SEKTION_REVIEW_ESCALATION:
                self._log(
                    f"  ⚠️  Sektion {sektion['id']}: {ergebnis.review_quote:.0%} Review-Quote → Eskalation empfohlen"
                )

            # Disputed-Befunde-Eskalation
            disputed_count = sum(
                1 for b in ergebnis.befunde if b.bewertung.value == "disputed"
            )
            if disputed_count > 0:
                logger.warning(
                    "⚠️  Sektion %s: %d strittige Befunde – manuelle Eskalation empfohlen",
                    sektion["id"],
                    disputed_count,
                )

            sektionsergebnisse.append(ergebnis)

            # Checkpoint: Zwischenergebnis sichern
            self._save_checkpoint(
                sektionsergebnisse,
                checkpoint_dir,
                review_budget=self.review_budget,
                review_markierte_felder=review_markierte_felder,
                review_budget_erreicht=review_budget_erreicht,
            )
            if review_budget_erreicht:
                break

        if gepruefte_felder == 0:
            raise ValueError(
                "Keine Prüffelder wurden verarbeitet. "
                "Bitte --sektionen prüfen oder einen Katalog mit gültigen Prüffeldern verwenden."
            )

        # ── Schritt 4: Berichte generieren ───────────────────────────────
        self._log("\n📝 Schritt 4/4: Prüfberichte generieren")
        generator = BerichtGenerator(
            institution=self.institution,
            pruefer=f"FinRegAgents v2.0 – {label}",
            regulatorik=self.regulatorik,
            model=self.model,
            katalog_version=katalog_version,
        )
        stats_file, costs = self._write_run_stats()
        report_paths = generator.generiere_alle_berichte(
            sektionsergebnisse=sektionsergebnisse,
            output_dir=self.output_dir,
            token_stats=self._token_stats_summary(stats_file, costs),
            stats_file=stats_file,
            verbose=self.verbose_token_details,
        )
        if self.evidence_relevance_filter:
            report_paths["relevance_filter_report"] = (
                self._write_relevance_filter_report(agent)
            )

        # ── Zusammenfassung ──────────────────────────────────────────────
        t_total = time.time() - t_start
        self._log(f"\n{'=' * 60}")
        self._log(f"✅ Prüfung abgeschlossen in {t_total:.0f}s")
        self._log(f"   Regulatorik: {label}")
        self._log(f"   Prüffelder:  {gepruefte_felder}/{total_felder}")
        if self.review_budget is not None:
            self._log(
                f"   Review-Budget: {review_markierte_felder}/{self.review_budget} "
                f"(erreicht={review_budget_erreicht})"
            )
        self._log("   Berichte:")
        for fmt, pth in report_paths.items():
            self._log(f"     {fmt.upper()}: {pth}")

        return report_paths

    def _write_relevance_filter_report(self, agent: PrueferAgent) -> str:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "relevance_filter_samples.json"
        drops = list(agent.relevance_filter_drops)
        sample = random.sample(drops, k=min(20, len(drops))) if drops else []
        payload = {
            "enabled": True,
            "stats": dict(agent.relevance_filter_stats),
            "sample_size": len(sample),
            "dropped_samples": sample,
            "klassifikation": [
                "regulatory_requirement",
                "control_evidence",
                "context_noise",
            ],
            "drop_reasons": [
                "NO_REG_REF",
                "MARKETING_PHRASE",
                "NON_CONTROL_CONTEXT",
            ],
        }
        out_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(out_path)

    def _save_checkpoint(
        self,
        sektionsergebnisse: list,
        checkpoint_dir: Path,
        review_budget: int | None = None,
        review_markierte_felder: int = 0,
        review_budget_erreicht: bool = False,
    ):
        """Sichert Zwischenergebnisse nach jeder Sektion."""
        try:
            data = []
            for s in sektionsergebnisse:
                data.append(
                    {
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
                        ],
                    }
                )
            path = checkpoint_dir / "checkpoint_latest.json"
            path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            if review_budget is not None:
                meta_path = checkpoint_dir / "checkpoint_meta.json"
                meta_payload = {
                    "review_budget": review_budget,
                    "review_markierte_felder": review_markierte_felder,
                    "review_budget_erreicht": review_budget_erreicht,
                    "fortsetzung_erforderlich": review_budget_erreicht,
                }
                meta_path.write_text(
                    json.dumps(meta_payload, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
        except Exception as e:
            logger.warning("Checkpoint-Fehler (Pipeline läuft weiter): %s", e)

    def _load_completed_sektion_ids(self, checkpoint_dir: Path) -> set[str]:
        """Loads already-completed section IDs from the latest checkpoint."""
        checkpoint_path = checkpoint_dir / "checkpoint_latest.json"
        if not checkpoint_path.exists():
            logger.warning("Keine Checkpoint-Datei gefunden unter: %s", checkpoint_path)
            return set()
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            return {entry["id"] for entry in data if "id" in entry}
        except Exception as e:
            logger.warning("Fehler beim Laden des Checkpoints: %s", e)
            return set()

    def _add_token_usage(self, agent_name: str, usage: dict):
        input_tokens = int((usage or {}).get("input", 0))
        output_tokens = int((usage or {}).get("output", 0))
        total_tokens = int((usage or {}).get("total", input_tokens + output_tokens))

        agent_bucket = self.run_token_stats["nach_agent"].setdefault(
            agent_name, {"input": 0, "output": 0, "total": 0}
        )
        agent_bucket["input"] += input_tokens
        agent_bucket["output"] += output_tokens
        agent_bucket["total"] += total_tokens
        self.run_token_stats["gesamt"]["input"] += input_tokens
        self.run_token_stats["gesamt"]["output"] += output_tokens
        self.run_token_stats["gesamt"]["total"] += total_tokens

    def _add_detail_stat(self, sektion_id: str, prueffeld_id: str, befund):
        self.run_token_stats["details"].append(
            {
                "sektion": sektion_id,
                "prueffeld": prueffeld_id,
                "confidence": befund.confidence,
                "confidence_level": befund.confidence_level,
                "review_erforderlich": befund.review_erforderlich,
                "token_usage": befund.token_usage,
                "confidence_guards": befund.confidence_guards,
            }
        )

    def _estimate_costs(self) -> dict:
        # V1-Schätzung: konservatives Dummy-Pricing pro 1k Tokens.
        pricing = {
            "pruefer": {"input_per_1k": 0.003, "output_per_1k": 0.015},
            "skeptiker": {"input_per_1k": 0.003, "output_per_1k": 0.015},
            "currency": "USD",
        }
        details = {}
        total_cost = 0.0
        for agent_name, usage in self.run_token_stats["nach_agent"].items():
            rates = pricing.get(agent_name, pricing["pruefer"])
            in_cost = (usage["input"] / 1000.0) * rates["input_per_1k"]
            out_cost = (usage["output"] / 1000.0) * rates["output_per_1k"]
            agent_cost = round(in_cost + out_cost, 6)
            details[agent_name] = {
                "input_cost": round(in_cost, 6),
                "output_cost": round(out_cost, 6),
                "total_cost": agent_cost,
            }
            total_cost += agent_cost
        return {
            "currency": pricing["currency"],
            "total_cost": round(total_cost, 6),
            "nach_agent": details,
            "pricing_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _write_run_stats(self) -> tuple[str, dict]:
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / "run_stats.json"
        costs = self._estimate_costs()
        payload = {
            "token_stats": {
                "version": self.run_token_stats["version"],
                "gesamt": self.run_token_stats["gesamt"],
                "nach_agent": self.run_token_stats["nach_agent"],
            },
            "kosten_schaetzung": costs,
            "stats_file": str(stats_path),
        }
        if self.verbose_token_details:
            payload["details"] = self.run_token_stats["details"]

        stats_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return str(stats_path), costs

    def _token_stats_summary(self, stats_file: str, costs: dict) -> dict:
        return {
            "version": self.run_token_stats["version"],
            "gesamt": self.run_token_stats["gesamt"],
            "nach_agent": self.run_token_stats["nach_agent"],
            "kosten_schaetzung": {
                "currency": costs["currency"],
                "total_cost": costs["total_cost"],
                "pricing_timestamp": costs["pricing_timestamp"],
            },
            "stats_file": stats_file,
        }

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
  python pipeline.py --input ./docs --regulatorik wphg --provider openai --model gpt-4o
  python pipeline.py --input ./docs --regulatorik gwg --provider gemini --model gemini-2.0-flash
  python pipeline.py --input ./docs --regulatorik gwg --provider ollama --model llama3.3
  python pipeline.py --input ./docs --regulatorik gwg --provider mistral --embedding-provider mistral
        """,
    )
    parser.add_argument(
        "--input", required=True, help="Verzeichnis mit Prüfungsdokumenten"
    )
    parser.add_argument(
        "--institution", default="Prüfinstitut", help="Name des Instituts"
    )
    parser.add_argument(
        "--regulatorik",
        default="gwg",
        choices=list(KATALOG_REGISTRY.keys()),
        help="Zu prüfende Regulatorik",
    )
    parser.add_argument(
        "--output", default="./reports/output", help="Ausgabeverzeichnis"
    )
    parser.add_argument(
        "--catalog", default=None, help="Eigener Katalog (überschreibt --regulatorik)"
    )
    parser.add_argument(
        "--provider",
        default="anthropic",
        choices=list_providers(),
        help=(
            "LLM-Provider (Default: anthropic). "
            f"Verfügbar: {', '.join(list_providers())}"
        ),
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Modellname (Default: Provider-spezifisch, z.B. claude-sonnet-4-6 / gpt-4o / gemini-2.0-flash)",
    )
    parser.add_argument(
        "--embedding-provider",
        default=None,
        dest="embedding_provider",
        choices=list_embedding_providers() + [None],
        help=(
            "Embedding-Provider (Default: auto – openai wenn Key vorhanden, sonst fastembed). "
            f"Verfügbar: {', '.join(list_embedding_providers())}"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        dest="embedding_model",
        help="Embedding-Modellname (Default: Provider-spezifisch)",
    )
    parser.add_argument(
        "--sektionen", nargs="*", help="Nur diese Sektionen prüfen (z.B. S01 S02)"
    )
    parser.add_argument("--top-k", type=int, default=8, help="RAG-Chunks pro Prüffrage")
    parser.add_argument(
        "--review-budget",
        type=int,
        default=None,
        help="Stoppt den Lauf nach N review-markierten Befunden und schreibt einen Checkpoint.",
    )
    parser.add_argument(
        "--evidence-relevance-filter",
        action="store_true",
        default=False,
        help="Aktiviert Spike-Preprocessor: klassifiziert Chunks und droppt context_noise.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Ausführliche Token-Detailstatistik in Reports und run_stats.json",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        default=False,
        help="Unterdrückt Fortschrittsausgabe auf der Konsole",
    )
    parser.add_argument(
        "--skeptiker",
        action="store_true",
        default=False,
        help="Skeptiker-Agent aktivieren (adversariales Review)",
    )
    parser.add_argument(
        "--skeptiker-only-konform",
        action="store_true",
        default=False,
        help="Skeptiker nur für 'konform'-Ratings aktivieren",
    )
    parser.add_argument(
        "--adversarial",
        action="store_true",
        default=False,
        help="Adversarial Prompting Layer: zweiter LLM-Pass mit umgekehrtem "
        "System-Prompt auf gleicher Evidenz. Große Abweichung → "
        "Confidence-Penalty + Review-Markierung.",
    )
    parser.add_argument(
        "--local-embeddings",
        action="store_true",
        default=False,
        help="Lokale Embeddings (FastEmbed, kein OpenAI-Key nötig). "
        "Wird automatisch aktiviert wenn OPENAI_API_KEY fehlt.",
    )
    parser.add_argument(
        "--relevance-filter",
        action="store_true",
        default=False,
        dest="relevance_filter",
        help="Relevanz-Filter aktivieren: context_noise Chunks werden vor "
        "LLM-Auswertung herausgefiltert. Schreibt relevance_sampling.json.",
    )
    parser.add_argument(
        "--review-budget",
        type=int,
        default=0,
        dest="review_budget",
        help="Pausiert die Prüfung nach N Befunden mit review_erforderlich=True und speichert einen Checkpoint (0 = deaktiviert).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Setzt eine pausierte Prüfung fort (liest den letzten Checkpoint und überspringt bereits abgeschlossene Sektionen).",
    )
    args = parser.parse_args()

    pipeline = AuditPipeline(
        input_dir=args.input,
        institution=args.institution,
        regulatorik=args.regulatorik,
        catalog_path=args.catalog,
        output_dir=args.output,
        provider=args.provider,
        model=args.model,
        embedding_provider=args.embedding_provider,
        embedding_model=args.embedding_model,
        sektionen_filter=args.sektionen,
        top_k=args.top_k,
        verbose=not args.quiet,
        verbose_token_details=args.verbose,
        review_budget=args.review_budget,
        evidence_relevance_filter=args.evidence_relevance_filter,
        skeptiker=args.skeptiker,
        skeptiker_only_konform=args.skeptiker_only_konform,
        adversarial=args.adversarial,
        use_relevance_filter=args.relevance_filter,
        resume=args.resume,
    )
    try:
        pipeline.run()
    except ReviewBudgetExceeded as exc:
        print(f"\n⏸️  {exc}")
        sys.exit(0)


if __name__ == "__main__":
    main()
