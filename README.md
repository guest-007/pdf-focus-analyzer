# PDF Focus Analyzer

A map-reduce pipeline that analyzes PDFs through a user-defined lens. Give it a PDF and a focus prompt — it extracts, chunks, retrieves, maps claims with evidence, and reduces everything into a structured report.

Requires **OpenAI API** or **LM Studio** (local models via OpenAI-compatible API).

This is a **learning project** exploring how far **Qwen 3.5-35B** can go for focused document analysis, compared to OpenAI models.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-65%20passed-brightgreen.svg)](#tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## What It Does

You write a short focus prompt (e.g. "SWOT analysis of this company") and point it at a PDF. The pipeline produces a Markdown report containing:

- **Summary** — a narrative overview, written in the language of your input
- **Key findings** — numbered, prioritized insights
- **Subtheme analysis** — the LLM breaks your focus into subthemes and analyzes each
- **Evidence table** — every claim linked to a direct quote and page number(s)
- **Contradictions and gaps** — where the document is inconsistent or silent
- **Confidence score** — how well the source material covers the focus

All claims trace back to specific pages. The output language matches the input.

---

## Providers

The pipeline uses the OpenAI client for both cloud and local models. You need one of:

| Provider | Chat Model | Embedding | Setup |
|----------|-----------|-----------|-------|
| **OpenAI** | Any OpenAI model (default: gpt-4.1) | text-embedding-3-small | Set `OPENAI_API_KEY` in `.env` |
| **LM Studio** | Any local model | nomic-embed-text-v1.5 | Run LM Studio with models loaded |

Both providers use the same pipeline. Token budgets adapt automatically — 128K context for OpenAI, 4K for local models.

To change models, edit [`infra/config.py`](infra/config.py).

---

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For OpenAI, create a `.env` file:

```bash
OPENAI_API_KEY=sk-...
```

Place your PDF in `input/`, create your focus prompt:

```bash
cp input/focus_example.md input/focus.md
# edit focus.md with your analysis focus
```

Run:

```bash
python analyze_pdf.py
```

The CLI will ask you to pick a PDF and a provider. Or pass everything explicitly:

```bash
python analyze_pdf.py --pdf input/report.pdf --focus input/focus.md --provider openai
```

---

## How It Works

1. **Parse focus** — LLM interprets your prompt into a structured spec with subthemes, keywords, and retrieval queries
2. **Extract + chunk** — PDF text is extracted page-by-page, then split into overlapping token-aware chunks
3. **Embed + retrieve** — chunks are embedded (FAISS), multi-query retrieval selects the top-k most relevant
4. **Map** — each retrieved chunk is analyzed: claims extracted with quotes, page refs, and evidence strength
5. **Reduce** — two-phase synthesis (batch groups of 5, then merge) produces the final structured summary
6. **Report** — Markdown report written to `out/`

Key design choices:
- **Structured JSON output** — all LLM calls enforce JSON schemas (Pydantic + OpenAI strict mode)
- **Dynamic token budgets** — each stage computes available output tokens from context limit minus input size
- **Two-phase reduce** — handles large documents without exceeding context limits

---

## CLI Reference

```
usage: analyze_pdf.py [--pdf PDF] [--focus FOCUS] [--provider {openai,lmstudio}]
                      [--top-k TOP_K] [--out DIR]

Options:
  --pdf              Path to PDF (default: interactive picker from input/)
  --focus            Focus prompt or .md file (default: input/focus.md)
  --provider         openai or lmstudio (default: interactive picker)
  --top-k            Number of chunks to retrieve (default: 20)
  --out              Output directory (default: out)
```

---

## Tests

```bash
python -m pytest tests/ -v
```

65 unit tests covering models, config, chunking, JSON parsing, orchestration, and CLI.

---

## License

[MIT](LICENSE)
