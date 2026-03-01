# 🔍 PDF Focus Analyzer

> **Analyze any PDF through a custom lens.** Define what you care about, and get a structured, evidence-backed report — powered by LLMs.

This is a **learning project** where I evaluate how far I can push **Qwen 3.5-35B** (and compare it with OpenAI models) for focused PDF analysis.

Instead of a generic "chat with PDF" flow, this repo aims for a reproducible pipeline:
**extract → chunk → retrieve → map claims/evidence → reduce summary → challenge pass**.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-66%20passed-brightgreen.svg)](#-tests)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)

---

## 🧪 Project Status

- **Status:** Active learning experiment
- **Current focus:** Evaluate Qwen 3.5-35B quality, instruction following, and latency
- **Constraint:** 4096-token context experiments for local models
- **Output style:** Structured, evidence-traceable analysis (not free-form chat)

---

## ✨ What It Does

Drop in a PDF and a **focus prompt** describing what you want to analyze. The pipeline will:

1. 📄 **Extract** text from the PDF page by page
2. ✂️ **Chunk** text into overlapping segments with token awareness
3. 🔎 **Retrieve** the most relevant chunks using vector embeddings (FAISS)
4. 🗺️ **Map** — extract structured claims and evidence from each chunk
5. 🔗 **Reduce** — synthesize findings in a two-phase batch + merge process
6. ⚖️ **Challenge** — spot-check overlooked chunks for missed evidence
7. 📊 **Report** — produce a Markdown report with findings, evidence table, confidence score, and gaps

All LLM outputs use **structured JSON output** (Pydantic models + OpenAI strict mode) for reliable parsing.

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

### 2. Set up your API key

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=sk-...
```

### 3. Add your PDF and focus

Place your PDF in the `input/` folder, then create your focus file by copying the example:

```bash
cp input/focus_example.md input/focus.md
```

Edit `input/focus.md` with your own analysis focus:

```markdown
# Focus: Risk Culture

Analyze the company's risk culture, covering:

- Governance and oversight model
- Leadership attitudes toward risk-taking
- Communication and escalation practices
```

(See [`input/focus_example.md`](input/focus_example.md) for a full example.)

### 4. Run it

```bash
python analyze_pdf.py
```

That's it! The interactive CLI will ask you to:
- **Pick a PDF** from `input/`
- **Pick a provider** (OpenAI or LM Studio)

Or pass everything explicitly:

```bash
python analyze_pdf.py --pdf input/report.pdf --focus input/focus.md --provider openai
```

---

## 📁 Project Structure

```
pdf-focus-analyzer/
├── analyze_pdf.py              # 🎯 CLI entry point
├── input/
│   └── focus_example.md        # Example focus prompt (copy to focus.md)
├── pipeline/
│   ├── pdf_extract.py          # Stage 2: PDF → pages
│   ├── chunking.py             # Stage 3: Pages → overlapping chunks
│   ├── retrieval.py            # Stage 4: FAISS vector search
│   ├── focus_parser.py         # Stage 1: Focus prompt → structured spec
│   ├── map_extract.py          # Stage 5: Chunk → claims extraction
│   ├── reduce_summarize.py     # Stage 6: Two-phase reduce synthesis
│   ├── quality_check.py        # Stage 7: Challenge pass
│   └── orchestrator.py         # Full pipeline orchestration
├── infra/
│   ├── config.py               # ⚙️ Model & provider configuration
│   ├── chat_factory.py         # LLM provider abstraction
│   ├── models.py               # Pydantic data models
│   ├── llm_json.py             # JSON response parsing
│   └── tokens.py               # Token budget utilities
├── tests/                      # 66 unit tests
├── requirements.txt
└── out/                        # Generated reports (gitignored)
    ├── report_*.md             # 📊 Final reports
    ├── final_result.json       # Structured result
    └── intermediate/           # Stage-by-stage JSON artifacts
```

---

## ⚙️ Configuration

All model settings live in [`infra/config.py`](infra/config.py). Change models, context limits, or timeouts in one place:

```python
@dataclass(frozen=True)
class AppConfig:
    openai_chat: ChatModelConfig       # model_name, context_limit, max_output_tokens
    lmstudio_chat: ChatModelConfig     # same, for local models
    openai_embedding: EmbeddingModelConfig
    lmstudio_embedding: EmbeddingModelConfig
    openai: ProviderConfig             # base_url (None = SDK default)
    lmstudio: ProviderConfig           # base_url for local server
```

### 🤖 Supported Providers

| Provider | Default Chat Model | Embedding Model | Notes |
|----------|-------------------|-----------------|-------|
| **OpenAI** | gpt-4.1 (128K context) | text-embedding-3-small | Any OpenAI model works (gpt-4.1, o3, etc.) |
| **LM Studio** | Any local model | nomic-embed-text-v1.5 | Requires LM Studio running locally |

> **Any OpenAI chat model** that supports structured output can be used — just change the model name and context limit in the config. The pipeline dynamically adapts token budgets to the model's context window.

To use a **different model**, just edit `infra/config.py`:

```python
openai_chat: ChatModelConfig = field(
    default_factory=lambda: ChatModelConfig(
        model_name="gpt-4.1-mini",      # ← change model here
        context_limit=128_000,
        max_output_tokens=32_768,
    )
)
```

---

## 🧪 Tests

```bash
python -m pytest tests/ -v
```

66 unit tests covering models, config, chunking, JSON parsing, orchestration, and CLI.

---

## 🏗️ How the Pipeline Works

```
┌─────────────┐    ┌──────────┐    ┌──────────┐    ┌───────────┐
│  Focus .md  │───→│  Parse   │───→│ Retrieval│───→│  Extract  │
│  + PDF      │    │  Prompt  │    │  Queries │    │  Pages    │
└─────────────┘    └──────────┘    └──────────┘    └───────────┘
                                                         │
                   ┌──────────┐    ┌────────────┐  ┌─────▼─────┐
                   │Challenge │◄───│   Reduce   │◄─│   Chunk   │
                   │  Pass    │    │  (2-phase) │  │  + Embed  │
                   └────┬─────┘    └────────────┘  └───────────┘
                        │               ▲
                        │          ┌────┴─────┐
                        └─────────→│  Report  │
                                   │  (.md)   │
                                   └──────────┘
```

### 🔑 Key Design Decisions

- **Dynamic token budgets** — Each pipeline stage computes how much output it can request based on the model's context limit minus the input size. Works seamlessly across 128K (OpenAI) and 4K (local models).
- **Input truncation** — When input exceeds the context window, it's automatically truncated to fit, with minimum output budgets preserved.
- **Two-phase reduce** — Map results are batched (5 per group), each batch synthesized, then merged into a final summary. This handles large documents without exceeding context limits.
- **Structured output** — All LLM calls use JSON schema enforcement (OpenAI strict mode) with Pydantic models, making parsing reliable.
- **Challenge pass** — A random sample of low-scoring chunks is re-examined to catch evidence the retrieval stage might have missed.

---

## 📋 CLI Reference

```
usage: analyze_pdf.py [--pdf PDF] [--focus FOCUS] [--provider {openai,lmstudio}]
                      [--top-k TOP_K] [--challenge-sample N] [--out DIR]

Options:
  --pdf              Path to PDF (default: interactive picker from input/)
  --focus            Focus prompt or .md file (default: input/focus.md)
  --provider         openai or lmstudio (default: interactive picker)
  --top-k            Number of chunks to retrieve (default: 20)
  --challenge-sample Challenge pass sample size (default: 5)
  --out              Output directory (default: out)
```

---

## 📜 License

[MIT](LICENSE)
