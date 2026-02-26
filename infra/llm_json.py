import re
from typing import Type, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def parse_json_response(raw: str, model_class: Type[T]) -> T:
    """Extract JSON from an LLM response and validate against a Pydantic model.

    Handles: pure JSON, markdown-fenced JSON, JSON embedded in text.
    """
    cleaned = raw.strip()

    # Try direct parse
    try:
        return model_class.model_validate_json(cleaned)
    except Exception:
        pass

    # Extract from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", cleaned, re.DOTALL)
    if fence_match:
        try:
            return model_class.model_validate_json(fence_match.group(1).strip())
        except Exception:
            pass

    # Find first balanced JSON object or array
    json_str = _extract_json_object(cleaned)
    if json_str:
        try:
            return model_class.model_validate_json(json_str)
        except Exception:
            pass

    raise ValueError(
        f"Could not parse valid {model_class.__name__} from LLM response.\n"
        f"Raw response (first 500 chars): {raw[:500]}"
    )


def _extract_json_object(text: str) -> str | None:
    """Find the first balanced JSON object or array in text."""
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            c = text[i]
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if c == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c == start_char:
                depth += 1
            elif c == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None
