"""Build concise user-visible summaries for Kimi reasoning."""

from collections.abc import Iterable
import json
import re


_MAX_SUMMARY_CHARS = 240
_JUSTIFICATION_PATTERN = re.compile(
    r"\bjustification\s*:\s*(\"(?:\\.|[^\"\\])*\")",
)


def _trim_summary(text: str, *, keep_tail: bool = False) -> str:
    normalized = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if len(normalized) <= _MAX_SUMMARY_CHARS:
        return normalized
    if keep_tail:
        return "…" + normalized[-(_MAX_SUMMARY_CHARS - 1):]
    return normalized[:_MAX_SUMMARY_CHARS - 1] + "…"


def _justification_from_text(text: str) -> str:
    try:
        value = json.loads(text)
    except (TypeError, ValueError, json.JSONDecodeError):
        value = None

    if isinstance(value, dict):
        justification = value.get("justification")
        if isinstance(justification, str) and justification.strip():
            return _trim_summary(justification)
        custom_input = value.get("input")
        if isinstance(custom_input, str):
            text = custom_input

    match = _JUSTIFICATION_PATTERN.search(text)
    if match is None:
        return ""
    try:
        justification = json.loads(match.group(1))
    except (TypeError, ValueError, json.JSONDecodeError):
        return ""
    return _trim_summary(justification) if isinstance(justification, str) else ""


def build_visible_reasoning_summary(
    reasoning_text: str,
    tool_calls: Iterable[dict] = (),
) -> str:
    """Prefer an approval justification, otherwise expose only the last two reasoning lines."""
    for tool_call in tool_calls:
        function = tool_call.get("function")
        arguments = (
            function.get("arguments")
            if isinstance(function, dict)
            else tool_call.get("arguments")
        )
        if isinstance(arguments, str):
            justification = _justification_from_text(arguments)
            if justification:
                return justification

        custom_input = tool_call.get("custom_input")
        if isinstance(custom_input, str):
            justification = _justification_from_text(custom_input)
            if justification:
                return justification

    lines = [line.strip() for line in reasoning_text.splitlines() if line.strip()]
    if not lines:
        return ""
    return _trim_summary("\n".join(lines[-2:]), keep_tail=True)
