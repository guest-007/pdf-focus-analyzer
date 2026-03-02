from tenacity import retry, stop_after_attempt, retry_if_exception_type

from infra.chat_factory import ChatModel, system_message, human_message, make_response_format
from infra.models import FocusSpec
from infra.llm_json import parse_json_response
from infra.tokens import compute_output_budget

FOCUS_PARSE_SYSTEM_PROMPT = """\
You are an expert document analyst. Given a user's focus prompt,
produce a structured analysis specification.

Rules:
- Rewrite the user's focus into a clear, unambiguous primary_focus.
- Generate 4-8 subthemes that decompose the focus into concrete dimensions.
- Generate both broad and narrow retrieval_queries (at least 5).
- Include relevant keywords and their synonyms (including German equivalents if appropriate).
- inclusion_criteria: what makes a text chunk relevant.
- exclusion_criteria: what to skip (e.g., boilerplate disclaimers, table of contents).
- Fill in ALL fields with real values.
- Do NOT fabricate anything. This is a planning step only.
- Respond in the same language as the user's focus prompt.
"""


@retry(
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(ValueError),
)
def parse_focus_prompt(focus_prompt: str, llm: ChatModel) -> FocusSpec:
    """Parse a free-text focus prompt into a structured FocusSpec."""
    msgs = [
        system_message(FOCUS_PARSE_SYSTEM_PROMPT),
        human_message(f"User focus prompt:\n\n{focus_prompt}"),
    ]
    budget = compute_output_budget(llm.context_limit, msgs, max_output=llm.max_output_tokens)
    raw = llm.generate(
        msgs,
        response_format=make_response_format(FocusSpec),
        max_tokens=budget,
    )
    return parse_json_response(raw, FocusSpec)
