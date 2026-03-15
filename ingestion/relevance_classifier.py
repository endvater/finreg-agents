"""
FinRegAgents – Evidence Relevance Classifier (Issue #29)

Klassifiziert Dokument-Chunks in drei Kategorien:
  - regulatory_requirement: Text enthält regulatorische Anforderungen
  - control_evidence: Text enthält Kontroll-Evidenz
  - context_noise: Text ist irrelevant (wird gefiltert)

Feature-Flag-gesteuert über AuditPipeline(use_relevance_filter=True).
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Enums und Datenklassen
# ------------------------------------------------------------------ #


class ChunkCategory(str, Enum):
    REGULATORY_REQUIREMENT = "regulatory_requirement"
    CONTROL_EVIDENCE = "control_evidence"
    CONTEXT_NOISE = "context_noise"


# Reason codes for filtered chunks
DROP_REASONS = {
    "NO_REG_REF": "Kein regulatorischer Bezug",
    "MARKETING_PHRASE": "Marketingsprache erkannt",
    "NON_CONTROL_CONTEXT": "Kein Kontrollkontext",
}

# Safety: never drop chunks containing these patterns
REGULATORY_ANCHORS = [
    r"§\s*\d+",
    r"Art\.\s*\d+",
    r"MaRisk",
    r"DORA",
    r"GwG",
    r"WpHG",
    r"MaComp",
    r"BaFin",
    r"KWG",
    r"AMLA",
]

# Control evidence signals: dates, percentages, policy references, action verbs
CONTROL_EVIDENCE_PATTERNS = [
    r"\d{2}\.\d{2}\.\d{4}",  # German date format
    r"\d{4}-\d{2}-\d{2}",  # ISO date format
    r"\d+\s*%",  # Percentages
    r"wurde\s+durchgeführt",
    r"liegt\s+vor",
    r"implementiert",
    r"dokumentiert",
    r"geprüft",
    r"festgestellt",
    r"nachgewiesen",
    r"Richtlinie",
    r"Policy",
    r"Prozess",
    r"Handbuch",
    r"Kontroll",
    r"Audit",
    r"Nachweis",
    r"Protokoll",
]

# Noise signals: marketing language, navigation, boilerplate
NOISE_PATTERNS = [
    r"Willkommen",
    r"Impressum",
    r"Copyright\s*©?",
    r"Seite\s+\d+",
    r"Table\s+of\s+Contents",
    r"Inhaltsverzeichnis",
    r"Alle\s+Rechte\s+vorbehalten",
    r"www\.",
    r"http[s]?://",
    r"info@",
]


@dataclass
class ClassifiedChunk:
    node_id: str
    text: str
    category: ChunkCategory
    drop_reason: str | None = None  # only set if context_noise


# ------------------------------------------------------------------ #
# Klassifizierer
# ------------------------------------------------------------------ #


class EvidenceRelevanceClassifier:
    """
    Regel-basierter Klassifizierer für Dokument-Chunks.
    Kein extra LLM-Call – läuft vollständig lokal.
    """

    def __init__(self):
        # Pre-compile patterns for performance
        self._anchor_patterns = [
            re.compile(p, re.IGNORECASE) for p in REGULATORY_ANCHORS
        ]
        self._evidence_patterns = [
            re.compile(p, re.IGNORECASE) for p in CONTROL_EVIDENCE_PATTERNS
        ]
        self._noise_patterns = [re.compile(p, re.IGNORECASE) for p in NOISE_PATTERNS]

    def classify(self, text: str) -> tuple[ChunkCategory, str | None]:
        """
        Regel-basierte Klassifizierung.
        Returns (category, drop_reason_code|None).

        Klassifizierungsreihenfolge (Priorität):
        1. Regulatory anchor → regulatory_requirement (nie noise)
        2. Control evidence signals → control_evidence
        3. Noise signals oder zu kurz → context_noise
        4. Default → control_evidence (sicher)
        """
        # Safety check: never classify as noise if regulatory anchor present
        for pattern in self._anchor_patterns:
            if pattern.search(text):
                return ChunkCategory.REGULATORY_REQUIREMENT, None

        # Check for control evidence signals
        for pattern in self._evidence_patterns:
            if pattern.search(text):
                return ChunkCategory.CONTROL_EVIDENCE, None

        # Check for noise signals
        for pattern in self._noise_patterns:
            if pattern.search(text):
                return ChunkCategory.CONTEXT_NOISE, "MARKETING_PHRASE"

        # Short texts with no regulatory content → noise
        if len(text.strip()) < 100:
            return ChunkCategory.CONTEXT_NOISE, "NO_REG_REF"

        # Default: treat as control evidence (safe fallback)
        return ChunkCategory.CONTROL_EVIDENCE, None

    def filter_chunks(self, nodes: list, regulatorik: str) -> tuple[list, list]:
        """
        Klassifiziert alle Nodes und filtert context_noise heraus.

        Args:
            nodes: Liste von LlamaIndex-Documents oder beliebigen Objekten
                   mit .text oder .get_content() Attribut
            regulatorik: Aktive Regulatorik (für Logging)

        Returns:
            (kept_nodes, dropped_nodes_with_reasons)
            dropped_nodes_with_reasons ist eine Liste von ClassifiedChunk
            mit category=CONTEXT_NOISE
        """
        kept = []
        dropped = []

        for i, node in enumerate(nodes):
            # Unterstützt Document (.text) und TextNode (.get_content())
            if hasattr(node, "text"):
                text = node.text
            elif hasattr(node, "get_content"):
                text = node.get_content()
            else:
                text = str(node)

            node_id = (
                getattr(node, "doc_id", None)
                or getattr(node, "id_", None)
                or f"chunk_{i}"
            )

            category, drop_reason = self.classify(text)

            if category == ChunkCategory.CONTEXT_NOISE:
                classified = ClassifiedChunk(
                    node_id=str(node_id),
                    text=text[:200],  # Truncate for report
                    category=category,
                    drop_reason=drop_reason,
                )
                dropped.append(classified)
                logger.debug(
                    "Chunk %s gefiltert [%s]: %s...",
                    node_id,
                    drop_reason,
                    text[:80],
                )
            else:
                kept.append(node)

        logger.info(
            "Relevanz-Filter [%s]: %d Chunks behalten, %d als context_noise gefiltert",
            regulatorik,
            len(kept),
            len(dropped),
        )
        return kept, dropped
