import json
import time
from datetime import datetime
from pathlib import Path

from infra.chat_factory import ChatFactory, ChatModel, EmbeddingModel
from infra.models import (
    FocusSpec,
    PageDoc,
    Chunk,
    RetrievedChunk,
    ChunkFocusResult,
    FinalSummary,
    ChallengeResult,
)
from pipeline.focus_parser import parse_focus_prompt
from pipeline.pdf_extract import extract_pages
from pipeline.chunking import chunk_pages
from pipeline.retrieval import ChunkIndex
from pipeline.map_extract import extract_focus_claims
from pipeline.reduce_summarize import reduce_results
from pipeline.quality_check import challenge_low_score_chunks


def save_json(data, path: Path) -> None:
    """Save Pydantic models or dicts to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(data, "model_dump"):
        obj = data.model_dump()
    elif isinstance(data, list) and data and hasattr(data[0], "model_dump"):
        obj = [item.model_dump() for item in data]
    else:
        obj = data
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def render_markdown(summary: FinalSummary) -> str:
    """Render a FinalSummary as a readable Markdown report."""
    lines: list[str] = []

    lines.append("# Focused Analysis Report\n")

    # Summary
    lines.append("## Summary\n")
    lines.append(summary.focused_summary)
    lines.append("")

    # Key Findings
    lines.append("## Key Findings\n")
    for i, finding in enumerate(summary.key_findings, 1):
        lines.append(f"{i}. {finding}")
    lines.append("")

    # Subtheme Analysis
    lines.append("## Subtheme Analysis\n")
    for st in summary.subtheme_synthesis:
        lines.append(f"### {st.subtheme}\n")
        lines.append(st.summary)
        lines.append("")

    # Evidence
    lines.append("## Evidence\n")
    lines.append("| Claim | Quote | Pages | Strength |")
    lines.append("|-------|-------|-------|----------|")
    for ev in summary.evidence_items:
        claim = ev.claim.replace("|", "\\|")
        quote = ev.quote.replace("|", "\\|").replace("\n", " ")
        pages = ", ".join(str(p) for p in ev.pages)
        lines.append(f"| {claim} | {quote} | {pages} | {ev.strength} |")
    lines.append("")

    # Contradictions
    lines.append("## Contradictions\n")
    if summary.contradictions:
        for c in summary.contradictions:
            lines.append(f"- {c}")
    else:
        lines.append("No contradictions found.")
    lines.append("")

    # Gaps
    lines.append("## Gaps\n")
    if summary.gaps:
        for g in summary.gaps:
            lines.append(f"- {g}")
    else:
        lines.append("No gaps identified.")
    lines.append("")

    # Confidence
    lines.append("## Confidence\n")
    lines.append(
        f"**Score: {summary.confidence.score:.2f}** — {summary.confidence.why}"
    )
    lines.append("")

    return "\n".join(lines)


def run_pipeline(
    pdf_path: str,
    focus_prompt: str,
    provider: str = "openai",
    top_k: int = 30,
    challenge_sample: int = 10,
    output_dir: str = "out",
) -> FinalSummary:
    """Run the full focused PDF analysis pipeline."""
    out = Path(output_dir)
    factory = ChatFactory()

    if provider == "lmstudio":
        factory.ensure_lms_ready()

    return _run_pipeline_stages(
        factory, provider, pdf_path, focus_prompt, top_k, challenge_sample, out
    )


def _run_pipeline_stages(
    factory: ChatFactory,
    provider: str,
    pdf_path: str,
    focus_prompt: str,
    top_k: int,
    challenge_sample: int,
    out: Path,
) -> FinalSummary:
    """Execute all pipeline stages. Separated so finally-block can clean up."""
    llm: ChatModel = factory.create(provider)
    embedder: EmbeddingModel = factory.create_embedder(provider)
    intermediate = out / "intermediate"

    print(f"\n{'=' * 60}")
    print("FOCUSED PDF ANALYSIS PIPELINE")
    print(f"PDF: {pdf_path}")
    print(f"Focus: {focus_prompt[:80]}{'...' if len(focus_prompt) > 80 else ''}")
    print(f"Provider: {provider}")
    print(f"{'=' * 60}\n")

    t0 = time.time()

    # Stage 1: Parse focus prompt
    print("[1/8] Parsing focus prompt...")
    focus_spec: FocusSpec = parse_focus_prompt(focus_prompt, llm)
    save_json(focus_spec, intermediate / "01_focus_spec.json")
    print(
        f"  -> {len(focus_spec.retrieval_queries)} retrieval queries, "
        f"{len(focus_spec.subthemes)} subthemes"
    )

    # Stage 2: Extract PDF pages
    print("[2/8] Extracting PDF pages...")
    pages: list[PageDoc] = extract_pages(pdf_path)
    save_json(pages, intermediate / "02_pages.json")
    print(f"  -> {len(pages)} pages extracted")

    # Stage 3: Chunk pages
    print("[3/8] Chunking text...")
    chunks: list[Chunk] = chunk_pages(pages)
    save_json(chunks, intermediate / "03_chunks.json")
    print(f"  -> {len(chunks)} chunks created")

    # Stage 4: Embed and retrieve
    print("[4/8] Building embeddings and retrieving relevant chunks...")
    index = ChunkIndex(embedder)
    index.build(chunks)
    retrieved: list[RetrievedChunk] = index.retrieve(focus_spec, top_k=top_k)
    save_json(retrieved, intermediate / "04_retrieved.json")
    print(f"  -> {len(retrieved)} chunks retrieved (top-k={top_k})")

    # Stage 5: Map extraction
    print("[5/8] Running focused extraction on retrieved chunks...")
    map_results: list[ChunkFocusResult] = extract_focus_claims(
        retrieved, focus_spec, llm
    )
    save_json(map_results, intermediate / "05_map_results.json")
    relevant_count = sum(1 for r in map_results if r.focus_relevance_score > 0.3)
    print(f"  -> {relevant_count}/{len(map_results)} chunks scored as relevant")

    # Stage 6: Reduce synthesis (two-phase: batch reduce → final merge)
    print("[6/8] Synthesizing final summary (two-phase reduce)...")
    summary: FinalSummary = reduce_results(map_results, focus_spec, llm)
    save_json(summary, intermediate / "06_summary.json")
    print(f"  -> Confidence: {summary.confidence.score:.2f}")

    # Stage 7: Challenge pass
    print("[7/8] Running challenge pass on low-score chunks...")
    retrieved_ids = {r.chunk_id for r in retrieved}
    challenge: ChallengeResult = challenge_low_score_chunks(
        chunks,
        retrieved_ids,
        focus_spec,
        summary,
        llm,
        sample_size=challenge_sample,
    )
    save_json(challenge, intermediate / "07_challenge.json")
    print(
        f"  -> Sampled {challenge.chunks_sampled} chunks, "
        f"contradictions found: {challenge.contradictions_found}"
    )

    # Stage 8: Merge challenge updates into final result
    if challenge.contradictions_found and challenge.material_updates:
        summary.contradictions.extend(challenge.material_updates)
        summary.evidence_items.extend(challenge.revised_evidence)
        summary.confidence.why += f" [Challenge pass: {challenge.audit_note}]"

    save_json(summary, out / "final_result.json")

    # Stage 9: Render markdown report
    print("[8/8] Writing markdown report...")
    report_md = render_markdown(summary)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = out / f"report_{timestamp}_{provider}.md"
    report_path.write_text(report_md, encoding="utf-8")

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"DONE in {elapsed:.1f}s")
    print(f"Intermediates: {intermediate}/")
    print(f"Report: {report_path}")
    print(f"{'=' * 60}\n")

    return summary
