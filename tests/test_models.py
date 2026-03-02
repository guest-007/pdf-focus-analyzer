"""Tests for infra.models -- all Pydantic data models."""

import pytest
from pydantic import ValidationError

from infra.models import (
    EvidenceRequirements,
    OutputFormat,
    FocusSpec,
    PageDoc,
    Chunk,
    RetrievedChunk,
    Claim,
    ChunkFocusResult,
    BatchSynthesis,
    SubthemeSynthesis,
    EvidenceItem,
    ConfidenceScore,
    FinalSummary,
)


class TestFocusSpec:
    def test_minimal(self):
        spec = FocusSpec(
            primary_focus="risk",
            analysis_goal="analyze risk",
            subthemes=["governance"],
            keywords=["risk"],
            synonyms=["Risiko"],
            inclusion_criteria=["mentions risk"],
            exclusion_criteria=["table of contents"],
            retrieval_queries=["risk culture"],
        )
        assert spec.primary_focus == "risk"
        assert spec.evidence_requirements.require_page_citations is True
        assert spec.output_format.style == "detailed"

    def test_defaults(self):
        spec = FocusSpec(
            primary_focus="x",
            analysis_goal="x",
            subthemes=[],
            keywords=[],
            synonyms=[],
            inclusion_criteria=[],
            exclusion_criteria=[],
            retrieval_queries=[],
        )
        assert spec.evidence_requirements.max_quote_length_chars == 300
        assert "key_findings" in spec.output_format.sections


class TestPageDoc:
    def test_creation(self):
        p = PageDoc(pdf_id="report", page=1, text="Hello world")
        assert p.page == 1


class TestChunk:
    def test_token_count_default(self):
        c = Chunk(pdf_id="r", chunk_id="r_0001", text="abc", start_page=1, end_page=1)
        assert c.token_count == 0

    def test_with_token_count(self):
        c = Chunk(
            pdf_id="r",
            chunk_id="r_0001",
            text="abc",
            start_page=1,
            end_page=2,
            token_count=100,
        )
        assert c.token_count == 100


class TestRetrievedChunk:
    def test_creation(self):
        rc = RetrievedChunk(
            chunk_id="c1",
            score=0.95,
            text="text",
            start_page=1,
            end_page=1,
        )
        assert rc.score == 0.95


class TestClaim:
    def test_creation(self):
        c = Claim(
            claim="Bank has policy",
            evidence_quote="Policy states...",
            page_refs=[5, 6],
            evidence_strength="strong",
        )
        assert c.evidence_strength == "strong"


class TestChunkFocusResult:
    def test_empty_claims(self):
        r = ChunkFocusResult(
            chunk_id="c1",
            focus_relevance_score=0.0,
            subtheme_hits=[],
            claims=[],
            risks_or_concerns=[],
            uncertainties=[],
        )
        assert r.focus_relevance_score == 0.0


class TestBatchSynthesis:
    def test_creation(self):
        bs = BatchSynthesis(
            key_findings=["Finding 1"],
            evidence_items=[
                EvidenceItem(claim="c", quote="q", pages=[1], strength="medium"),
            ],
            subtheme_hits=["governance"],
            contradictions=[],
            gaps=["gap1"],
        )
        assert len(bs.key_findings) == 1
        assert bs.subtheme_hits == ["governance"]


class TestFinalSummary:
    def test_full(self):
        s = FinalSummary(
            focused_summary="Summary text",
            key_findings=["f1", "f2"],
            subtheme_synthesis=[SubthemeSynthesis(subtheme="Gov", summary="s")],
            evidence_items=[
                EvidenceItem(claim="c", quote="q", pages=[1], strength="strong"),
            ],
            contradictions=[],
            gaps=["Gap 1"],
            confidence=ConfidenceScore(score=0.8, why="good"),
        )
        assert len(s.key_findings) == 2
        assert s.confidence.score == 0.8


