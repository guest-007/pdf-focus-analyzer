"""Token estimation utilities for context budget management."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from infra.chat_factory import ChatMessage


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for mixed Latin/German text."""
    return len(text) // 4


def estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    """Estimate total tokens in a list of chat messages."""
    total = 0
    for msg in messages:
        total += estimate_tokens(msg["content"]) + 4  # role + framing overhead
    return total


def compute_output_budget(
    context_limit: int,
    messages: list[ChatMessage],
    min_output: int = 200,
    max_output: int = 32_768,
) -> int:
    """Compute how many output tokens remain after the input messages.

    Capped by *max_output* (API completion-token limit, e.g. 32768 for OpenAI).
    Returns at least *min_output* so the model can always produce something.
    """
    input_tokens = estimate_messages_tokens(messages)
    remaining = context_limit - input_tokens
    budget = max(remaining, min_output)
    return min(budget, max_output)


def truncate_text_to_budget(text: str, token_budget: int) -> str:
    """Truncate text to fit within a token budget (by char estimate)."""
    max_chars = token_budget * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated to fit context ...]"
