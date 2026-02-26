import random

from tenacity import retry, stop_after_attempt, retry_if_exception_type

from infra.chat_factory import ChatModel, system_message, human_message, make_response_format
from infra.models import FocusSpec, Chunk, FinalSummary, ChallengeResult
from infra.llm_json import parse_json_response
from infra.tokens import estimate_tokens, compute_output_budget

CHALLENGE_SYSTEM_PROMPT = """\
Check if unseen chunks contradict or supplement these findings.
Focus: {primary_focus}
Current findings:
{key_findings}

Rules:
- If nothing material found, set contradictions_found to false and leave lists empty.
- revised_evidence items must include verbatim quotes and page references.
- audit_note: one-sentence summary of what you found.
"""

MIN_OUTPUT_TOKENS = 500


@retry(stop=stop_after_attempt(2), retry=retry_if_exception_type(ValueError))
def challenge_low_score_chunks(
    all_chunks: list[Chunk],
    retrieved_ids: set[str],
    focus_spec: FocusSpec,
    current_summary: FinalSummary,
    llm: ChatModel,
    sample_size: int = 10,
    seed: int = 42,
) -> ChallengeResult:
    """Sample non-retrieved chunks and check for contradictions."""
    non_retrieved = [c for c in all_chunks if c.chunk_id not in retrieved_ids]

    if not non_retrieved:
        return ChallengeResult(
            chunks_sampled=0,
            contradictions_found=False,
            material_updates=[],
            revised_evidence=[],
            audit_note="No non-retrieved chunks available for challenge pass.",
        )

    # Sample: half random, half keyword-triggered
    rng = random.Random(seed)
    keyword_lower = [k.lower() for k in focus_spec.keywords + focus_spec.synonyms]

    keyword_chunks = [
        c for c in non_retrieved if any(kw in c.text.lower() for kw in keyword_lower)
    ]
    random_chunks = rng.sample(
        non_retrieved,
        min(sample_size // 2, len(non_retrieved)),
    )
    keyword_sample = (
        rng.sample(
            keyword_chunks,
            min(sample_size - len(random_chunks), len(keyword_chunks)),
        )
        if keyword_chunks
        else []
    )

    # Combine and deduplicate
    challenge_set_ids: set[str] = set()
    challenge_set: list[Chunk] = []
    for c in keyword_sample + random_chunks:
        if c.chunk_id not in challenge_set_ids:
            challenge_set_ids.add(c.chunk_id)
            challenge_set.append(c)

    if not challenge_set:
        return ChallengeResult(
            chunks_sampled=0,
            contradictions_found=False,
            material_updates=[],
            revised_evidence=[],
            audit_note="No chunks selected for challenge pass.",
        )

    key_findings = "\n".join(f"- {f}" for f in current_summary.key_findings)
    sys_content = CHALLENGE_SYSTEM_PROMPT.format(
        primary_focus=focus_spec.primary_focus,
        key_findings=key_findings,
    )

    # Build chunks text, truncating to fit context budget
    prompt_overhead = estimate_tokens(sys_content) + 50  # message framing
    available_for_chunks = llm.context_limit - prompt_overhead - MIN_OUTPUT_TOKENS

    chunks_text = ""
    included = 0
    for c in challenge_set:
        entry = f"Chunk {c.chunk_id} (pages {c.start_page}-{c.end_page}):\n{c.text}\n\n---\n\n"
        if estimate_tokens(chunks_text + entry) > available_for_chunks:
            break
        chunks_text += entry
        included += 1

    sys = system_message(sys_content)
    user_msg = human_message(
        f"UNSEEN CHUNKS ({included} chunks):\n\n{chunks_text}"
    )
    msgs = [sys, user_msg]
    budget = compute_output_budget(llm.context_limit, msgs, min_output=MIN_OUTPUT_TOKENS, max_output=llm.max_output_tokens)
    raw = llm.generate(
        msgs,
        response_format=make_response_format(ChallengeResult),
        max_tokens=budget,
    )
    result = parse_json_response(raw, ChallengeResult)
    result.chunks_sampled = included
    return result
