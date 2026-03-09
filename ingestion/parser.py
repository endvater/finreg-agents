import re
from typing import List, Dict, Any

from llama_index.core.schema import TextNode
from llama_index.core.node_parser import SentenceSplitter


class RegulatoryParser:
    """
    Parser for specific German regulatory texts (e.g., GwG, DORA, MaRisk).
    Splits text by structural markers (e.g., Articles, Paragraphs, Chapters, Modules) 
    instead of arbitrary sentences, and injects structural metadata into the nodes.
    """

    def __init__(self, fallback_chunk_size: int = 1000, fallback_chunk_overlap: int = 100):
        self.fallback_chunk_size = fallback_chunk_size
        self._fallback_splitter = SentenceSplitter(
            chunk_size=fallback_chunk_size, 
            chunk_overlap=fallback_chunk_overlap
        )
        
        # Regex patterns to detect structure boundaries.
        # This matches:
        # - "Art. 1", "Artikel 1"
        # - "§ 1", "§1"
        # - "Tz. 1", "Textziffer 1" (MaRisk specific)
        # - "Modul AT 1", "BTR 3" (MaRisk specific modules)
        # - "(1)", "(2)" (Paragraphs/Absätze) - these are subclasses usually
        
        self.patterns = {
            "chapter": re.compile(r"^(?:Kapitel|Teil|Abschnitt)\s+(?:[IVX]+|\d+)(?=\s*[:.\s])", re.MULTILINE | re.IGNORECASE),
            "module": re.compile(r"^(?:Modul\s+)?(AT|BTR|BTO|BTK)\s+(\d+(?:\.\d+)*)\s*", re.MULTILINE),
            "article": re.compile(r"^(?:Art\.|Artikel|Article|Recital)\s+(\d+[a-z]*)", re.MULTILINE | re.IGNORECASE),
            "paragraph": re.compile(r"^§\s*(\d+[a-z]*)", re.MULTILINE),
            "margin_no": re.compile(r"^(?:Tz\.|Textziffer)\s*(\d+)", re.MULTILINE | re.IGNORECASE),
            "sub_paragraph": re.compile(r"^\s*\(\s*(\d+[a-z]*)\s*\)(?=\s+\w{3,})", re.MULTILINE)
        }

    def parse_text(self, text: str, base_metadata: Dict[str, Any] = None) -> List[TextNode]:
        """
        Parses a full regulatory text string and returns a list of enriched LlamaIndex TextNodes.
        """
        if not base_metadata:
            base_metadata = {}

        # 1. Identify all structural markers and their positions
        markers = self._find_markers(text)

        # 2. Split text into chunks based on these markers
        if not markers:
            # Fallback for plain text without clear regulatory structure
            return self._fallback_split(text, base_metadata)
            
        nodes = self._create_nodes_from_markers(text, markers, base_metadata)
        
        # 3. Handle excessively large chunks 
        final_nodes = []
        for node in nodes:
            if len(node.text) > self.fallback_chunk_size * 2:
                final_nodes.extend(self._fallback_split(node.text, node.metadata))
            else:
                final_nodes.append(node)
                
        return final_nodes

    def _find_markers(self, text: str) -> List[Dict]:
        """Extracts positions and types of structural dividers."""
        markers = []
        for level, pattern in self.patterns.items():
            for match in pattern.finditer(text):
                
                # Handling multi-group extractions correctly (Bug 3)
                if level == "module" and len(match.groups()) >= 2:
                    extracted_id = f"{match.group(1)} {match.group(2)}"
                elif len(match.groups()) > 0 and match.group(1) is not None:
                    extracted_id = match.group(1)
                else:
                    extracted_id = match.group(0).strip()
                    
                markers.append({
                    "level": level,
                    "value": match.group(0).strip(),
                    "extracted_id": extracted_id,
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Sort sequentially by character index
        markers.sort(key=lambda x: int(x["start"]))
        
        # Filter out noisy/overlapping markers
        filtered_markers: List[Dict[str, Any]] = []
        for i, m in enumerate(markers):
            if i == 0 or int(m["start"]) >= int(filtered_markers[-1]["end"]):
                filtered_markers.append(m)
                
        return filtered_markers

    def _create_nodes_from_markers(self, text: str, markers: List[Dict], base_metadata: Dict) -> List[TextNode]:
        """Slices the text and constructs metadata for each block."""
        nodes = []
        context_state = {
            "chapter": None,
            "module": None,
            "article": None,
            "paragraph": None,
            "margin_no": None,
            "sub_paragraph": None
        }

        # Handle text before the very first marker (e.g. Title, Preamble)
        if markers and markers[0]["start"] > 0:
            pre_text = text[0:markers[0]["start"]].strip()
            if len(pre_text) > 20: # ignore whitespace
                meta = base_metadata.copy()
                meta["hierarchy_level"] = "Preamble/Title"
                nodes.append(TextNode(text=pre_text, metadata=meta))

        for i, marker in enumerate(markers):
            start_idx = int(marker["start"])
            end_idx = int(markers[i+1]["start"]) if i + 1 < len(markers) else len(text)
            
            chunk_text = text[start_idx:end_idx].strip()
            if len(chunk_text) < 10:
                continue

            # Update the running context
            level = marker["level"]
            context_state[level] = marker["extracted_id"]
            
            # Reset lower levels when a higher level shifts
            # Hierarchy roughly: Chapter > Module > Article/Paragraph > Margin No > Sub-paragraph
            if level in ["chapter", "module"]:
                context_state["article"] = None
                context_state["paragraph"] = None
                context_state["margin_no"] = None
                context_state["sub_paragraph"] = None
            elif level in ["article", "paragraph"]:
                context_state["margin_no"] = None
                context_state["sub_paragraph"] = None
            elif level == "margin_no":
                context_state["sub_paragraph"] = None

            # Build metadata block for this chunk
            chunk_meta = base_metadata.copy()
            chunk_meta["hierarchy_level"] = level
            chunk_meta["structural_id"] = marker["value"]
            
            # Add context trace
            for k, v in context_state.items():
                if v is not None:
                    chunk_meta[f"context_{k}"] = v
                    
            # Set the reference string (critical for Issue #6)
            ref_parts = []
            if context_state["chapter"]:
                ref_parts.append(f"{context_state['chapter']}")
            if context_state["module"]:
                ref_parts.append(f"Modul {context_state['module']}")
            if context_state["article"]:
                if "Recital" in str(context_state["article"]) or "Recital" in chunk_meta.get("structural_id", ""):
                    # Depending on how the ID was set, prefer structural_id if it's a Recital
                    if chunk_meta.get("structural_id", "").startswith("Recital"):
                        ref_parts.append(f"{chunk_meta['structural_id']}")
                    else:
                        ref_parts.append(f"Recital {context_state['article']}")
                else:
                    ref_parts.append(f"Art. {context_state['article']}")
            if context_state["paragraph"]:
                ref_parts.append(f"§ {context_state['paragraph']}")
            if context_state["margin_no"]:
                ref_parts.append(f"Tz. {context_state['margin_no']}")
            if context_state["sub_paragraph"]:
                ref_parts.append(f"Abs. {context_state['sub_paragraph']}")
            
            if ref_parts:
                chunk_meta["regulatory_reference"] = ", ".join(ref_parts)
            else:
                chunk_meta["regulatory_reference"] = marker["value"]

            nodes.append(TextNode(text=chunk_text, metadata=chunk_meta))

        return nodes

    def _fallback_split(self, text: str, base_metadata: Dict) -> List[TextNode]:
        """Naive fallback for text segments that are too long without structural markers."""
        from llama_index.core import Document
        
        baseline_doc = [Document(text=text, metadata=base_metadata)]
        nodes = self._fallback_splitter.get_nodes_from_documents(baseline_doc)
        
        # Ensure we don't return bare IndexNodes instead of TextNodes 
        return [TextNode(text=n.get_content(), metadata=n.metadata) for n in nodes]
