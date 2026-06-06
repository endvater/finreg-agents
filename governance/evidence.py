"""
Block C – EvidencePackage / Claim-Provenienz.

Erweitert die bestehende `agents.provenance.ClaimProvenance` (Korroborations-Status +
Chunk-IDs) um die im Buch geforderten Pflichtfelder: source_id, source_kind,
source_version, retrieved_at, method, quote_hash, confidence.

Reine Standardbibliothek – keine Abhängigkeit zu llama-index/langchain, damit das Modul
auch im Dashboard/CLI ohne Modell-Stack importierbar ist.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

_WS = re.compile(r"\s+")


def quote_hash(text: str) -> str:
    """Stabiler Hash einer zitierten Textstelle (Integritätsnachweis, Block M).

    Normalisiert (lowercase, Whitespace kollabiert, getrimmt), dann SHA-256.
    """
    norm = _WS.sub(" ", (text or "").strip().lower())
    return "sha256:" + hashlib.sha256(norm.encode("utf-8")).hexdigest()[:32]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ClaimStatus(str, Enum):
    CORROBORATED = "corroborated"  # >= 2 Quellen
    SINGLE_SOURCED = "single_sourced"  # genau 1 Quelle
    INFERRED = "inferred"  # vom Modell gefolgert, keine direkte Quelle
    UNVERIFIED = "unverified"  # keine Quelle gefunden
    DISPUTED = "disputed"  # widersprüchliche Quellen


class SourceKind(str, Enum):
    DOCUMENT = "document"  # ingestiertes Bankdokument
    CATALOG = "catalog"  # Prüfkatalog (Rechtsgrundlage)
    MODEL = "model"  # Modellwissen (kein externer Beleg)
    INTERVIEW = "interview"  # Interview-/Selbstauskunft
    UNKNOWN = "unknown"


@dataclass
class EvidenceClaim:
    """Ein einzelner belegter (oder explizit unbelegter) Claim mit voller Provenienz."""

    claim_text: str
    status: ClaimStatus = ClaimStatus.UNVERIFIED
    # Provenienz (Block C – Pflichtfelder)
    source_id: Optional[str] = None
    source_kind: SourceKind = SourceKind.UNKNOWN
    source_version: Optional[str] = None
    retrieved_at: Optional[str] = None
    method: str = "rag"  # rag | catalog | model_inference | term_check
    quote: Optional[str] = None
    quote_hash_value: Optional[str] = None
    confidence: float = 0.0
    source_chunk_ids: list[str] = field(default_factory=list)
    provenance_id: str = ""

    def __post_init__(self):
        if self.quote and not self.quote_hash_value:
            self.quote_hash_value = quote_hash(self.quote)
        if self.retrieved_at is None:
            self.retrieved_at = utc_now()

    @property
    def is_grounded(self) -> bool:
        """Belegt = auf eine reale Quelle zurückführbar (nicht inferiert/unverifiziert)."""
        return self.status in (ClaimStatus.CORROBORATED, ClaimStatus.SINGLE_SOURCED)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["source_kind"] = self.source_kind.value
        return d


@dataclass
class EvidencePackage:
    """Strukturierter Output eines Prüffelds: Claims + offene Fragen + verbotene Aktionen."""

    prueffeld_id: str
    claims: list[EvidenceClaim] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    prohibited_final_actions: list[str] = field(
        default_factory=lambda: ["file_report", "close_case", "notify_authority"]
    )
    created_at: str = field(default_factory=utc_now)

    @property
    def groundedness(self) -> float:
        """Anteil belegter Claims – Kern-Qualitätsmetrik (Block K)."""
        if not self.claims:
            return 0.0
        return sum(1 for c in self.claims if c.is_grounded) / len(self.claims)

    @property
    def has_unverified(self) -> bool:
        return any(
            c.status
            in (ClaimStatus.UNVERIFIED, ClaimStatus.INFERRED, ClaimStatus.DISPUTED)
            for c in self.claims
        )

    def to_dict(self) -> dict:
        return {
            "prueffeld_id": self.prueffeld_id,
            "claims": [c.to_dict() for c in self.claims],
            "open_questions": self.open_questions,
            "prohibited_final_actions": self.prohibited_final_actions,
            "created_at": self.created_at,
            "groundedness": round(self.groundedness, 4),
        }


def from_claim_provenance(
    prueffeld_id: str,
    provenance_list,
    *,
    catalog_version: Optional[str] = None,
    quotes_by_claim: Optional[dict] = None,
) -> EvidencePackage:
    """Bridge: bestehende `agents.provenance.ClaimProvenance` → EvidencePackage.

    Akzeptiert beliebige Objekte mit den Attributen claim_text/status/
    source_chunk_ids/provenance_id (Duck-Typing → keine harte Import-Kopplung).
    """
    status_map = {
        "corroborated": ClaimStatus.CORROBORATED,
        "single_sourced": ClaimStatus.SINGLE_SOURCED,
        "unverified": ClaimStatus.UNVERIFIED,
        "disputed": ClaimStatus.DISPUTED,
        "inferred": ClaimStatus.INFERRED,
    }
    quotes_by_claim = quotes_by_claim or {}
    claims: list[EvidenceClaim] = []
    for p in provenance_list or []:
        raw_status = getattr(
            getattr(p, "status", None), "value", getattr(p, "status", "unverified")
        )
        chunk_ids = list(getattr(p, "source_chunk_ids", []) or [])
        quote = quotes_by_claim.get(getattr(p, "provenance_id", ""))
        claims.append(
            EvidenceClaim(
                claim_text=getattr(p, "claim_text", ""),
                status=status_map.get(str(raw_status), ClaimStatus.UNVERIFIED),
                source_id=chunk_ids[0] if chunk_ids else None,
                source_kind=SourceKind.DOCUMENT if chunk_ids else SourceKind.MODEL,
                source_version=catalog_version,
                method="rag" if chunk_ids else "model_inference",
                quote=quote,
                source_chunk_ids=chunk_ids,
                provenance_id=getattr(p, "provenance_id", ""),
            )
        )
    return EvidencePackage(prueffeld_id=prueffeld_id, claims=claims)
