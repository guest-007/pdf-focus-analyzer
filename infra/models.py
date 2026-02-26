from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field

# --- Stage 1: Focus Parsing ---


class EvidenceRequirements(BaseModel):
    require_page_citations: bool = True
    require_quotes: bool = True
    max_quote_length_chars: int = 300


class OutputFormat(BaseModel):
    style: str = "detailed"
    sections: List[str] = Field(
        default_factory=lambda: ["key_findings", "evidence", "gaps", "confidence"]
    )


class FocusSpec(BaseModel):
    primary_focus: str
    analysis_goal: str
    subthemes: List[str]
    keywords: List[str]
    synonyms: List[str]
    inclusion_criteria: List[str]
    exclusion_criteria: List[str]
    evidence_requirements: EvidenceRequirements = Field(
        default_factory=EvidenceRequirements
    )
    retrieval_queries: List[str]
    output_format: OutputFormat = Field(default_factory=OutputFormat)


# --- Stage 2: PDF Extraction ---


class PageDoc(BaseModel):
    pdf_id: str
    page: int  # 1-based
    text: str


# --- Stage 3: Chunking ---


class Chunk(BaseModel):
    pdf_id: str
    chunk_id: str
    text: str
    start_page: int
    end_page: int
    token_count: int = 0


# --- Stage 4: Retrieval ---


class RetrievedChunk(BaseModel):
    chunk_id: str
    score: float
    text: str
    start_page: int
    end_page: int


# --- Stage 5: Map Step ---


class Claim(BaseModel):
    claim: str
    evidence_quote: str
    page_refs: List[int]
    evidence_strength: str  # "weak", "medium", "strong"


class ChunkFocusResult(BaseModel):
    chunk_id: str
    focus_relevance_score: float
    subtheme_hits: List[str]
    claims: List[Claim]
    risks_or_concerns: List[str]
    uncertainties: List[str]


# --- Stage 6a: Batch Reduce (intermediate) ---


class EvidenceItem(BaseModel):
    claim: str
    quote: str
    pages: List[int]
    strength: str


class BatchSynthesis(BaseModel):
    key_findings: List[str]
    evidence_items: List[EvidenceItem]
    subtheme_hits: List[str]
    contradictions: List[str]
    gaps: List[str]


# --- Stage 6b: Final Reduce ---


class SubthemeSynthesis(BaseModel):
    subtheme: str
    summary: str


class ConfidenceScore(BaseModel):
    score: float
    why: str


class FinalSummary(BaseModel):
    focused_summary: str
    key_findings: List[str]
    subtheme_synthesis: List[SubthemeSynthesis]
    evidence_items: List[EvidenceItem]
    contradictions: List[str]
    gaps: List[str]
    confidence: ConfidenceScore


# --- Stage 7: Quality Check ---


class ChallengeResult(BaseModel):
    chunks_sampled: int
    contradictions_found: bool
    material_updates: List[str]
    revised_evidence: List[EvidenceItem]
    audit_note: str
