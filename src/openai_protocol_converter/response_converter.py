"""Convert chat.completions responses to responses API format."""


def convert_response(chat_resp: dict) -> dict:
    """Convert a chat.completions response dict to responses API format."""
    choice = chat_resp["choices"][0]
    message = choice["message"]
    content = message.get("content", "")

    output_item: dict = {
        "type": "message",
        "role": message.get("role", "assistant"),
        "content": [{"type": "output_text", "text": content}],
    }

    # Handle usage mapping
    usage = chat_resp.get("usage", {})
    mapped_usage = {
        "input_tokens": usage.get("prompt_tokens", 0),
        "output_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }

    return {
        "id": chat_resp["id"],
        "object": "response",
        "created_at": chat_resp["created"],
        "model": chat_resp["model"],
        "output": [output_item],
        "usage": mapped_usage,
        "status": "completed",
    }
