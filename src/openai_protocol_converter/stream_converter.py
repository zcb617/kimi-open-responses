"""Convert chat.completions SSE stream to responses API SSE format."""
import json


class StreamConverter:
    """Converts chat.completions SSE events to responses API SSE events."""

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id
        self.model = model
        self._tool_calls: dict[int, dict] = {}
        self._emitted_tool_ids: set[int] = set()

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
        content = delta.get("content")
        tool_calls = delta.get("tool_calls")

        # Handle text content
        if content is not None:
            return json.dumps({
                "id": self.response_id,
                "output": [{"type": "output_text", "text": content}],
            }, separators=(",", ":"))

        # Handle tool_calls
        if tool_calls:
            return self._process_tool_call_delta(tool_calls)

        return None

    def _process_tool_call_delta(self, tool_calls: list[dict]) -> str | None:
        """Accumulate tool call deltas and emit when complete."""
        for tc in tool_calls:
            index = tc.get("index", 0)

            if index not in self._tool_calls:
                self._tool_calls[index] = {
                    "id": tc.get("id", ""),
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                }
            else:
                existing = self._tool_calls[index]
                if tc.get("id"):
                    existing["id"] = tc["id"]
                func = tc.get("function", {})
                if func.get("name"):
                    existing["name"] = func["name"]
                if func.get("arguments"):
                    existing["arguments"] += func["arguments"]

            existing = self._tool_calls[index]

            # Emit once we have an id and name (minimum viable tool call)
            if not existing["id"] or not existing["name"]:
                continue

            # Track that we've emitted this tool call
            if index not in self._emitted_tool_ids:
                self._emitted_tool_ids.add(index)

            return json.dumps({
                "id": self.response_id,
                "output": [{
                    "type": "output_function_call",
                    "call_id": existing["id"],
                    "name": existing["name"],
                    "arguments": existing["arguments"],
                }],
            }, separators=(",", ":"))

        return None
