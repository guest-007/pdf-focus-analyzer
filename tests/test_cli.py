"""Tests for analyze_pdf.py -- CLI focus reading."""

from pathlib import Path

from analyze_pdf import _read_focus


class TestReadFocus:
    def test_reads_md_file(self, tmp_path):
        md_file = tmp_path / "focus.md"
        md_file.write_text("# Risk Culture\n\nAnalyze risk.", encoding="utf-8")
        result = _read_focus(str(md_file))
        assert result == "# Risk Culture\n\nAnalyze risk."

    def test_inline_text(self):
        result = _read_focus("risk culture in banking")
        assert result == "risk culture in banking"

    def test_nonexistent_md_returns_string(self):
        result = _read_focus("nonexistent.md")
        assert result == "nonexistent.md"

    def test_non_md_file_returns_string(self, tmp_path):
        txt_file = tmp_path / "focus.txt"
        txt_file.write_text("content", encoding="utf-8")
        result = _read_focus(str(txt_file))
        assert result == str(txt_file)

    def test_strips_whitespace(self, tmp_path):
        md_file = tmp_path / "focus.md"
        md_file.write_text("  \n  content here  \n  ", encoding="utf-8")
        result = _read_focus(str(md_file))
        assert result == "content here"

    def test_real_input_file(self):
        result = _read_focus("input/focus_example.md")
        assert "Risk Culture" in result
        assert "Governance" in result
