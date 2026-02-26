"""Tests for infra.llm_json -- JSON parsing from LLM responses."""

import pytest

from infra.llm_json import parse_json_response, _extract_json_object
from infra.models import ConfidenceScore


class TestParseJsonResponse:
    def test_direct_json(self):
        raw = '{"score": 0.75, "why": "good coverage"}'
        result = parse_json_response(raw, ConfidenceScore)
        assert result.score == 0.75
        assert result.why == "good coverage"

    def test_markdown_fenced(self):
        raw = 'Here is the result:\n```json\n{"score": 0.5, "why": "partial"}\n```\n'
        result = parse_json_response(raw, ConfidenceScore)
        assert result.score == 0.5

    def test_fence_without_json_tag(self):
        raw = '```\n{"score": 0.9, "why": "strong"}\n```'
        result = parse_json_response(raw, ConfidenceScore)
        assert result.score == 0.9

    def test_embedded_in_text(self):
        raw = 'The answer is: {"score": 0.6, "why": "ok"} and that is all.'
        result = parse_json_response(raw, ConfidenceScore)
        assert result.score == 0.6

    def test_with_whitespace(self):
        raw = '  \n  {"score": 0.3, "why": "weak"}  \n  '
        result = parse_json_response(raw, ConfidenceScore)
        assert result.score == 0.3

    def test_invalid_raises_valueerror(self):
        with pytest.raises(ValueError, match="Could not parse"):
            parse_json_response("not json at all", ConfidenceScore)

    def test_wrong_schema_raises_valueerror(self):
        raw = '{"wrong_field": 123}'
        with pytest.raises(ValueError):
            parse_json_response(raw, ConfidenceScore)


class TestExtractJsonObject:
    def test_simple_object(self):
        assert _extract_json_object('{"a": 1}') == '{"a": 1}'

    def test_nested_object(self):
        text = 'prefix {"a": {"b": 2}} suffix'
        assert _extract_json_object(text) == '{"a": {"b": 2}}'

    def test_array(self):
        assert _extract_json_object("[1, 2, 3]") == "[1, 2, 3]"

    def test_no_json(self):
        assert _extract_json_object("no json here") is None

    def test_string_with_braces(self):
        text = '{"msg": "hello {world}"}'
        result = _extract_json_object(text)
        assert result == '{"msg": "hello {world}"}'

    def test_escaped_quotes(self):
        text = r'{"msg": "say \"hi\""}'
        result = _extract_json_object(text)
        assert result is not None
