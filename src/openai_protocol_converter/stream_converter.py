"""Convert chat.completions SSE stream to responses API SSE format."""
import json


class StreamConverter:
    """Converts chat.completions SSE events to responses API SSE events."""

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model

    def process_event(self, event_data: str) -> str | None:
        """Process a single chat.completions SSE event.

        Returns the converted responses API event string, or None to skip.
        """
        if event_data.strip() == "[DONE]":
            return json.dumps({"id": self.response_id, "status": "completed"}, separators=(",", ":"))

        try:
            data = json.loads(event_data)
        except json.JSONDecodeError:
            return None

        choices = data.get("choices", [])
        if not choices:
            return None

        delta = choices[0].get("delta", {})
        content = delta.get("content", "")

        if content is None or content == "":
            return None

        return json.dumps({
            "id": self.response_id,
            "output": [{"type": "output_text", "text": content}],
        }, separators=(",", ":"))
