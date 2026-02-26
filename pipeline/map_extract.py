from tenacity import retry, stop_after_attempt, retry_if_exception_type
from tqdm import tqdm

from infra.chat_factory import ChatModel, system_message, human_message, make_response_format
from infra.models import FocusSpec, RetrievedChunk, ChunkFocusResult
from infra.llm_json import parse_json_response
from infra.tokens import compute_output_budget

MAP_SYSTEM_PROMPT = """\
Extract claims relevant to: {primary_focus}
Goal: {analysis_goal}
Subthemes: {subthemes}

Rules:
- score 0.0-1.0. If irrelevant, score 0.0 and empty claims.
- evidence_quote must be copied verbatim from the chunk.
- evidence_strength: "weak", "medium", or "strong".
- Do NOT fabricate quotes or pages.
"""


@retry(stop=stop_after_attempt(2), retry=retry_if_exception_type(ValueError))
def _extract_single(
    chunk: RetrievedChunk, focus_spec: FocusSpec, llm: ChatModel
) -> ChunkFocusResult:
    sys = system_message(
        MAP_SYSTEM_PROMPT.format(
            primary_focus=focus_spec.primary_focus,
            analysis_goal=focus_spec.analysis_goal,
            subthemes=", ".join(focus_spec.subthemes),
        )
    )
    user_text = (
        f"Chunk ID: {chunk.chunk_id}\n"
        f"Pages: {chunk.start_page}-{chunk.end_page}\n\n"
        f"--- CHUNK TEXT ---\n{chunk.text}\n--- END CHUNK ---"
    )
    msgs = [sys, human_message(user_text)]
    budget = compute_output_budget(llm.context_limit, msgs, max_output=llm.max_output_tokens)
    raw = llm.generate(
        msgs,
        response_format=make_response_format(ChunkFocusResult),
        max_tokens=budget,
    )
    return parse_json_response(raw, ChunkFocusResult)


def extract_focus_claims(
    chunks: list[RetrievedChunk],
    focus_spec: FocusSpec,
    llm: ChatModel,
) -> list[ChunkFocusResult]:
    """Run map extraction on all retrieved chunks."""
    results = []
    for chunk in tqdm(chunks, desc="Map extraction"):
        try:
            result = _extract_single(chunk, focus_spec, llm)
            results.append(result)
        except Exception as e:
            print(f"  WARNING: Failed to extract from {chunk.chunk_id}: {e}")
            results.append(
                ChunkFocusResult(
                    chunk_id=chunk.chunk_id,
                    focus_relevance_score=0.0,
                    subtheme_hits=[],
                    claims=[],
                    risks_or_concerns=[],
                    uncertainties=[f"Extraction failed: {str(e)}"],
                )
            )
    return results
