"""Tests for pipeline.orchestrator -- save_json, render_markdown."""

import json
from pathlib import Path

import pytest

from infra.models import (
    FinalSummary,
    SubthemeSynthesis,
    EvidenceItem,
    ConfidenceScore,
)
from pipeline.orchestrator import save_json, render_markdown


def _make_summary(**overrides) -> FinalSummary:
    defaults = dict(
        focused_summary="Test summary.",
        key_findings=["Finding 1", "Finding 2"],
        subtheme_synthesis=[
            SubthemeSynthesis(subtheme="Governance", summary="Gov summary."),
        ],
        evidence_items=[
            EvidenceItem(
                claim="Claim A", quote="Quote A", pages=[1, 2], strength="strong"
            ),
        ],
        contradictions=[],
        gaps=["Gap 1"],
        confidence=ConfidenceScore(score=0.75, why="Good coverage"),
    )
    defaults.update(overrides)
    return FinalSummary(**defaults)


class TestSaveJson:
    def test_save_pydantic_model(self, tmp_path):
        summary = _make_summary()
        path = tmp_path / "out" / "result.json"
        save_json(summary, path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["focused_summary"] == "Test summary."

    def test_save_list_of_models(self, tmp_path):
        items = [
            EvidenceItem(claim="c1", quote="q1", pages=[1], strength="strong"),
            EvidenceItem(claim="c2", quote="q2", pages=[2], strength="weak"),
        ]
        path = tmp_path / "items.json"
        save_json(items, path)
        data = json.loads(path.read_text())
        assert len(data) == 2
        assert data[0]["claim"] == "c1"

    def test_save_dict(self, tmp_path):
        path = tmp_path / "raw.json"
        save_json({"key": "value"}, path)
        data = json.loads(path.read_text())
        assert data["key"] == "value"

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "a" / "b" / "c" / "file.json"
        save_json({"x": 1}, path)
        assert path.exists()


class TestRenderMarkdown:
    def test_contains_all_sections(self):
        md = render_markdown(_make_summary())
        assert "# Focused Analysis Report" in md
        assert "## Summary" in md
        assert "## Key Findings" in md
        assert "## Subtheme Analysis" in md
        assert "### Governance" in md
        assert "## Evidence" in md
        assert "## Contradictions" in md
        assert "## Gaps" in md
        assert "## Confidence" in md

    def test_key_findings_numbered(self):
        md = render_markdown(_make_summary())
        assert "1. Finding 1" in md
        assert "2. Finding 2" in md

    def test_evidence_table(self):
        md = render_markdown(_make_summary())
        assert "| Claim A |" in md
        assert "| Quote A |" in md
        assert "| 1, 2 |" in md
        assert "| strong |" in md

    def test_no_contradictions_message(self):
        md = render_markdown(_make_summary(contradictions=[]))
        assert "No contradictions found." in md

    def test_with_contradictions(self):
        md = render_markdown(_make_summary(contradictions=["Issue X"]))
        assert "- Issue X" in md

    def test_no_gaps_message(self):
        md = render_markdown(_make_summary(gaps=[]))
        assert "No gaps identified." in md

    def test_confidence_score(self):
        md = render_markdown(_make_summary())
        assert "**Score: 0.75**" in md
        assert "Good coverage" in md

    def test_pipe_in_evidence_escaped(self):
        s = _make_summary(
            evidence_items=[
                EvidenceItem(
                    claim="A | B", quote="Q | R", pages=[1], strength="medium"
                ),
            ]
        )
        md = render_markdown(s)
        assert "A \\| B" in md
        assert "Q \\| R" in md

    def test_empty_evidence(self):
        md = render_markdown(_make_summary(evidence_items=[]))
        assert "| Claim |" in md  # header still present

    def test_subtheme_content(self):
        md = render_markdown(_make_summary())
        assert "Gov summary." in md
