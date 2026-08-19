"""Convert chat.completions responses to responses API format."""
import uuid

from .reasoning_summary import build_visible_reasoning_summary


def _extract_text_content(content) -> str:
    """Normalize chat message content to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                text = part.get("text")
                if isinstance(text, str):
                    texts.append(text)
        return "".join(texts)
    return str(content)


def _build_usage(chat_usage: dict) -> dict:
    """Map Chat Completions usage to Responses usage shape."""
    return {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "input_tokens_details": {
            "cached_tokens": chat_usage.get("cached_tokens", 0),
        },
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens": chat_usage.get("reasoning_tokens", 0),
        },
        "total_tokens": chat_usage.get("total_tokens", 0),
    }


def convert_response(chat_resp: dict) -> dict:
    """Convert a chat.completions response dict to responses API format."""
    choice = chat_resp["choices"][0]
    message = choice.get("message", {})

    output_items: list[dict] = []
    message_id = f"msg_{uuid.uuid4().hex[:24]}"

    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content:
        summary_text = build_visible_reasoning_summary(
            reasoning_content,
            message.get("tool_calls") or [],
        )
        output_items.append({
            "id": f"rs_{uuid.uuid4().hex[:24]}",
            "type": "reasoning",
            "status": "completed",
            "summary": (
                [{"type": "summary_text", "text": summary_text}]
                if summary_text
                else []
            ),
            "content": [{
                "type": "reasoning_text",
                "text": reasoning_content,
            }],
        })

    refusal = message.get("refusal")
    tool_calls = message.get("tool_calls") or []
    content_text = _extract_text_content(message.get("content"))

    # Message item: keep refusal/text in standard message content.
    # Tool calls are emitted as standalone function_call output items.
    if refusal is not None:
        output_items.append({
            "id": message_id,
            "type": "message",
            "role": message.get("role", "assistant"),
            "status": "completed",
            "content": [{
                "type": "refusal",
                "refusal": str(refusal),
            }],
        })
    elif content_text or not tool_calls:
        output_items.append({
            "id": message_id,
            "type": "message",
            "role": message.get("role", "assistant"),
            "status": "completed",
            "content": [{
                "type": "output_text",
                "text": content_text,
                "annotations": [],
            }],
        })

    for idx, tool_call in enumerate(tool_calls):
        function = tool_call.get("function", {}) if isinstance(tool_call, dict) else {}
        call_id = ""
        if isinstance(tool_call, dict):
            call_id = tool_call.get("id", "") or ""
        if not call_id:
            call_id = f"call_{idx}"
        output_items.append({
            "id": f"fc_{call_id}",
            "type": "function_call",
            "status": "completed",
            "call_id": call_id,
            "name": function.get("name", "") if isinstance(function, dict) else "",
            "arguments": function.get("arguments", "") if isinstance(function, dict) else "",
        })

    return {
        "id": chat_resp["id"],
        "object": "response",
        "created_at": chat_resp["created"],
        "completed_at": chat_resp["created"],
        "model": chat_resp["model"],
        "status": "completed",
        "error": None,
        "incomplete_details": None,
        "instructions": None,
        "max_output_tokens": None,
        "output": output_items,
        "parallel_tool_calls": True,
        "previous_response_id": None,
        "tool_choice": "auto",
        "tools": [],
        "temperature": None,
        "top_p": None,
        "truncation": "disabled",
        "usage": _build_usage(chat_resp.get("usage", {})),
        "metadata": {},
    }
