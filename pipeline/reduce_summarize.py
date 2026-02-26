import json

from tenacity import retry, stop_after_attempt, retry_if_exception_type

from infra.chat_factory import ChatModel, system_message, human_message, make_response_format
from infra.models import FocusSpec, ChunkFocusResult, BatchSynthesis, FinalSummary
from infra.llm_json import parse_json_response
from infra.tokens import (
    estimate_tokens,
    compute_output_budget,
    truncate_text_to_budget,
)

BATCH_SIZE = 5

# Minimum output budget -- if the model has less room than this after input,
# the input gets truncated to make space.
MIN_BATCH_OUTPUT = 1200
MIN_FINAL_OUTPUT = 2000

BATCH_REDUCE_PROMPT = """\
Synthesize these evidence extractions into an intermediate summary.
Focus: {primary_focus}
Goal: {analysis_goal}
Subthemes: {subthemes}

Rules:
- key_findings: 2-4 most important findings from this batch.
- evidence_items: top claims with verbatim quotes and page references.
- subtheme_hits: which subthemes had evidence in this batch.
- Note any contradictions or gaps.
- Do NOT fabricate evidence.
"""

FINAL_REDUCE_PROMPT = """\
Merge these intermediate summaries into a final report.
Focus: {primary_focus}
Goal: {analysis_goal}
Subthemes: {subthemes}

Rules:
- focused_summary: 2-3 paragraph narrative synthesis.
- key_findings: 3-8 items, most important first. Deduplicate across batches.
- One subtheme_synthesis per subtheme, even if evidence is thin.
- evidence_items: top claims with verbatim quotes and page references.
- contradictions: where evidence conflicts.
- gaps: what the report does NOT address relative to the focus.
- confidence.score: 0.0 (no evidence) to 1.0 (comprehensive). Explain reasoning.
- Do NOT fabricate evidence.
"""


@retry(stop=stop_after_attempt(2), retry=retry_if_exception_type(ValueError))
def _reduce_batch(
    batch: list[ChunkFocusResult],
    focus_spec: FocusSpec,
    llm: ChatModel,
) -> BatchSynthesis:
    """Reduce a single batch of ChunkFocusResults into a BatchSynthesis."""
    sys = system_message(
        BATCH_REDUCE_PROMPT.format(
            primary_focus=focus_spec.primary_focus,
            analysis_goal=focus_spec.analysis_goal,
            subthemes=", ".join(focus_spec.subthemes),
        )
    )
    results_text = json.dumps(
        [r.model_dump() for r in batch], indent=1, ensure_ascii=False
    )
    # Truncate input if it would leave too little room for output
    sys_tokens = estimate_tokens(sys["content"]) + 4
    input_budget = llm.context_limit - sys_tokens - MIN_BATCH_OUTPUT
    results_text = truncate_text_to_budget(results_text, input_budget)

    user_msg = human_message(
        f"Evidence extractions ({len(batch)} chunks):\n\n{results_text}"
    )
    msgs = [sys, user_msg]
    budget = compute_output_budget(llm.context_limit, msgs, min_output=MIN_BATCH_OUTPUT, max_output=llm.max_output_tokens)
    raw = llm.generate(
        msgs,
        response_format=make_response_format(BatchSynthesis),
        max_tokens=budget,
    )
    return parse_json_response(raw, BatchSynthesis)


@retry(stop=stop_after_attempt(3), retry=retry_if_exception_type(ValueError))
def _final_synthesize(
    batches: list[BatchSynthesis],
    focus_spec: FocusSpec,
    llm: ChatModel,
) -> FinalSummary:
    """Merge batch syntheses into a final summary."""
    sys = system_message(
        FINAL_REDUCE_PROMPT.format(
            primary_focus=focus_spec.primary_focus,
            analysis_goal=focus_spec.analysis_goal,
            subthemes=", ".join(focus_spec.subthemes),
        )
    )
    batches_text = json.dumps(
        [b.model_dump() for b in batches], indent=1, ensure_ascii=False
    )
    # Truncate input if it would leave too little room for output
    sys_tokens = estimate_tokens(sys["content"]) + 4
    input_budget = llm.context_limit - sys_tokens - MIN_FINAL_OUTPUT
    batches_text = truncate_text_to_budget(batches_text, input_budget)

    user_msg = human_message(
        f"Intermediate summaries ({len(batches)} batches):\n\n{batches_text}\n\n"
        f"Produce the final synthesis."
    )
    msgs = [sys, user_msg]
    budget = compute_output_budget(llm.context_limit, msgs, min_output=MIN_FINAL_OUTPUT, max_output=llm.max_output_tokens)
    raw = llm.generate(
        msgs,
        response_format=make_response_format(FinalSummary),
        max_tokens=budget,
    )
    return parse_json_response(raw, FinalSummary)


def reduce_results(
    results: list[ChunkFocusResult],
    focus_spec: FocusSpec,
    llm: ChatModel,
    max_input_results: int = 20,
    batch_size: int = BATCH_SIZE,
) -> FinalSummary:
    """Two-phase reduce: batch synthesis -> final merge."""
    sorted_results = sorted(
        results, key=lambda r: r.focus_relevance_score, reverse=True
    )
    top_results = sorted_results[:max_input_results]

    # Phase 1: Batch reduce
    batches: list[BatchSynthesis] = []
    for i in range(0, len(top_results), batch_size):
        batch = top_results[i : i + batch_size]
        print(f"  Batch {len(batches) + 1}: {len(batch)} chunks...")
        batch_result = _reduce_batch(batch, focus_spec, llm)
        batches.append(batch_result)

    # Phase 2: Final synthesis
    print(f"  Final synthesis from {len(batches)} batches...")
    return _final_synthesize(batches, focus_spec, llm)
