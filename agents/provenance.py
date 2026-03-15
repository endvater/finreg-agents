"""
FinRegAgents – Per-Claim Provenance Tracking (Issue #6)

Annotates each sentence in a befund's Begründung with a corroboration status
based on keyword overlap with retrieved document chunks.
"""

from dataclasses import dataclass, field
from enum import Enum
import re

# German stopwords to filter out from keyword matching
_STOPWORDS = {
    "aber",
    "alle",
    "allem",
    "allen",
    "aller",
    "alles",
    "als",
    "also",
    "am",
    "an",
    "ander",
    "andere",
    "anderem",
    "anderen",
    "anderer",
    "anderes",
    "anderm",
    "andern",
    "anderr",
    "anders",
    "auch",
    "auf",
    "aus",
    "bei",
    "bin",
    "bis",
    "bist",
    "da",
    "damit",
    "dann",
    "das",
    "dass",
    "daß",
    "dass",
    "dazu",
    "dem",
    "den",
    "denn",
    "der",
    "des",
    "dessen",
    "die",
    "dies",
    "diese",
    "diesem",
    "diesen",
    "dieser",
    "dieses",
    "doch",
    "dort",
    "durch",
    "ein",
    "eine",
    "einem",
    "einen",
    "einer",
    "eines",
    "einige",
    "einigem",
    "einigen",
    "einiger",
    "einiges",
    "einmal",
    "er",
    "es",
    "etwas",
    "euch",
    "für",
    "gegen",
    "gewesen",
    "gibt",
    "hatte",
    "hatten",
    "hier",
    "hin",
    "hinter",
    "ich",
    "ihm",
    "ihn",
    "ihnen",
    "ihr",
    "ihre",
    "ihrem",
    "ihren",
    "ihrer",
    "ihres",
    "im",
    "in",
    "indem",
    "ins",
    "ist",
    "jede",
    "jedem",
    "jeden",
    "jeder",
    "jedes",
    "jetzt",
    "kann",
    "kein",
    "keine",
    "keinem",
    "keinen",
    "keiner",
    "keines",
    "können",
    "könnte",
    "machen",
    "manche",
    "manchem",
    "manchen",
    "mancher",
    "manches",
    "mein",
    "meine",
    "meinem",
    "meinen",
    "meiner",
    "meines",
    "mich",
    "mir",
    "mit",
    "muss",
    "müssen",
    "nach",
    "nachdem",
    "nicht",
    "nichts",
    "noch",
    "nun",
    "nur",
    "ob",
    "oder",
    "ohne",
    "sehr",
    "sein",
    "seine",
    "seinem",
    "seinen",
    "seiner",
    "seines",
    "selbst",
    "sich",
    "sie",
    "sind",
    "so",
    "solche",
    "solchem",
    "solchen",
    "solcher",
    "solches",
    "soll",
    "sollte",
    "sondern",
    "sonst",
    "sowie",
    "über",
    "um",
    "und",
    "uns",
    "unser",
    "unsere",
    "unserem",
    "unseren",
    "unserer",
    "unseres",
    "unter",
    "viel",
    "vom",
    "von",
    "vor",
    "war",
    "waren",
    "was",
    "weg",
    "weil",
    "weiter",
    "welche",
    "welchem",
    "welchen",
    "welcher",
    "welches",
    "wenn",
    "wer",
    "wie",
    "wieder",
    "wird",
    "wir",
    "wo",
    "wollen",
    "wollte",
    "würde",
    "würden",
    "zu",
    "zum",
    "zur",
    "zwar",
    "zwischen",
    # English stopwords (for mixed content)
    "the",
    "and",
    "for",
    "are",
    "but",
    "not",
    "you",
    "all",
    "can",
    "had",
    "her",
    "was",
    "one",
    "our",
    "out",
    "day",
    "get",
    "has",
    "him",
    "his",
    "how",
    "its",
    "may",
    "now",
    "say",
    "see",
    "who",
    "did",
    "she",
    "use",
    "way",
    "will",
    "with",
    "this",
    "that",
    "from",
    "they",
    "have",
    "been",
    "than",
    "when",
    "were",
    "what",
    "your",
    "which",
    "there",
    "their",
    "about",
    "would",
    "these",
    "other",
    "into",
    "more",
    "also",
    "some",
    "could",
    "than",
    "then",
    "them",
    "well",
    "said",
    "each",
    "just",
}

# Minimum keyword word length (> 3 chars per spec: "filter len>3")
_MIN_WORD_LEN = 3

# Minimum keyword overlap to consider a chunk as supporting a claim
_MIN_KEYWORD_OVERLAP = 3


class CorroborationStatus(str, Enum):
    CORROBORATED = "corroborated"  # 2+ chunks support it
    SINGLE_SOURCED = "single_sourced"  # exactly 1 chunk supports it
    UNVERIFIED = "unverified"  # no chunk found


@dataclass
class ClaimProvenance:
    claim_text: str  # the sentence/claim
    status: CorroborationStatus
    source_chunk_ids: list[str] = field(
        default_factory=list
    )  # which chunk node_ids support it
    provenance_id: str = ""  # short ID like "C001"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on '. ' and '.\n', filter short sentences."""
    # Split on period followed by space or newline
    parts = re.split(r"\.\s+|\.\n", text)
    sentences = []
    for part in parts:
        part = part.strip()
        if len(part) >= 20:
            sentences.append(part)
    return sentences


def _extract_keywords(sentence: str) -> set[str]:
    """
    Extract significant keywords from a sentence.
    Filters words to length > 3 chars and removes stopwords.
    """
    words = re.findall(r"[a-zA-ZäöüÄÖÜß]{4,}", sentence.lower())
    return {w for w in words if w not in _STOPWORDS}


def _get_chunk_id(node) -> str:
    """Get a stable ID for a retrieved chunk node."""
    try:
        node_id = node.node.node_id
        if node_id:
            return node_id
    except AttributeError:
        pass
    # Fallback: hash of first 50 chars of text
    try:
        text = node.node.text[:50]
        return f"hash_{abs(hash(text)):x}"
    except AttributeError:
        return f"hash_{abs(hash(str(node))):x}"


def _chunk_contains_keywords(chunk_text: str, keywords: set[str]) -> bool:
    """Check if chunk text contains >= _MIN_KEYWORD_OVERLAP of the given keywords."""
    if not keywords:
        return False
    chunk_lower = chunk_text.lower()
    overlap = sum(1 for kw in keywords if kw in chunk_lower)
    return overlap >= _MIN_KEYWORD_OVERLAP


def annotate_claims(
    befund_text: str,
    retrieved_nodes,  # list of NodeWithScore from LlamaIndex
) -> list[ClaimProvenance]:
    """
    Split befund_text into sentences.
    For each sentence, check how many retrieved chunks contain
    overlapping keywords (intersection of significant words, min 3 chars,
    filtered for stopwords).
    Returns list of ClaimProvenance.
    """
    sentences = _split_sentences(befund_text)
    result = []

    for i, sentence in enumerate(sentences):
        provenance_id = f"C{i + 1:03d}"
        keywords = _extract_keywords(sentence)

        supporting_chunk_ids: list[str] = []

        for node in retrieved_nodes:
            try:
                chunk_text = node.node.text
            except AttributeError:
                try:
                    chunk_text = node.get_content()
                except AttributeError:
                    chunk_text = str(node)

            if _chunk_contains_keywords(chunk_text, keywords):
                supporting_chunk_ids.append(_get_chunk_id(node))

        n_supporting = len(supporting_chunk_ids)
        if n_supporting >= 2:
            status = CorroborationStatus.CORROBORATED
        elif n_supporting == 1:
            status = CorroborationStatus.SINGLE_SOURCED
        else:
            status = CorroborationStatus.UNVERIFIED

        result.append(
            ClaimProvenance(
                claim_text=sentence,
                status=status,
                source_chunk_ids=supporting_chunk_ids,
                provenance_id=provenance_id,
            )
        )

    return result
