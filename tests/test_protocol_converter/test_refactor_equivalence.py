"""Behavior-lock tests for performance refactors."""
import copy

import pytest

from src.openai_protocol_converter.request_converter import _convert_content
from src.openai_protocol_converter.stream_converter import parse_sse_buffer


def _legacy_convert_content_part(part: dict) -> dict | None:
    part_type = part.get("type", "")
    if part_type == "input_text" or part_type == "output_text":
        return {"type": "text", "text": part.get("text", "")}
    if part_type == "input_image":
        image_url = part.get("image_url", "")
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
        if isinstance(image_url, dict):
            return {"type": "image_url", "image_url": image_url}
    if part_type == "refusal":
        return None
    if part_type in ("text", "image_url"):
        return part
    return None


def _legacy_convert_content(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        converted = []
        for part in content:
            if not isinstance(part, dict):
                continue
            cp = _legacy_convert_content_part(part)
            if cp:
                converted.append(cp)
        if converted and all(p.get("type") == "text" for p in converted):
            return "".join(p.get("text", "") for p in converted)
        if len(converted) == 1 and converted[0].get("type") == "text":
            return converted[0]["text"]
        return converted
    return content


def _legacy_parse_sse_buffer(buffer: str) -> tuple[list[dict], str]:
    events = []
    buffer = buffer.replace("\r\n", "\n")
    parts = buffer.split("\n\n")
    remaining = parts.pop() if parts else ""
    for part in parts:
        part = part.strip()
        if not part:
            continue
        event_data = ""
        for line in part.split("\n"):
            if line.startswith("data: "):
                event_data = line[6:]
            elif line.startswith("data:"):
                event_data = line[5:]
        if event_data:
            events.append({"data": event_data})
    return events, remaining


@pytest.mark.parametrize(
    "content",
    [
        "hello",
        "",
        [],
        [{}],
        [{"type": "input_text", "text": "a"}],
        [{"type": "output_text", "text": "b"}],
        [{"type": "text", "text": "x"}],
        [{"type": "text", "text": "x"}, {"type": "text", "text": "y"}],
        [{"type": "input_text", "text": "x"}, {"type": "output_text", "text": "y"}],
        [{"type": "input_image", "image_url": "https://a/b.png"}],
        [{"type": "image_url", "image_url": {"url": "https://a/b.png"}}],
        [{"type": "refusal", "refusal": "blocked"}],
        [{"type": "unknown", "v": 1}],
        [
            {"type": "input_text", "text": "a"},
            {"type": "input_image", "image_url": "https://a/b.png"},
            {"type": "refusal", "refusal": "r"},
            {"type": "text", "text": "b"},
            {"type": "image_url", "image_url": {"url": "https://a/c.png"}},
        ],
        ["not-a-dict", 1, None, {"type": "text", "text": "ok"}],
    ],
)
def test_convert_content_refactor_is_equivalent(content):
    before = _legacy_convert_content(copy.deepcopy(content))
    after = _convert_content(copy.deepcopy(content))
    assert after == before


@pytest.mark.parametrize(
    "buffer",
    [
        "",
        "data: hello\n\n",
        "data: hello\r\n\r\n",
        "data: hello\r\n\r\ndata: world\r\n\r\n",
        "event: message\ndata: hello\n\n",
        "data: first\ndata: second\n\n",
        "data: hello\n\ndata: wor",
        " data: spaced-leading\n\n",
        "\n\n",
        "data:\n\n",
        "data: [DONE]\n\n",
    ],
)
def test_parse_sse_buffer_refactor_is_equivalent(buffer):
    before_events, before_remaining = _legacy_parse_sse_buffer(buffer)
    after_events, after_remaining = parse_sse_buffer(buffer)
    assert after_events == before_events
    assert after_remaining == before_remaining
