"""Convert chat.completions SSE stream to responses API SSE format."""
import json
import time
import uuid


def parse_sse_buffer(buffer: str) -> tuple[list[dict], str]:
    """Parse an SSE buffer, returning (complete events, leftover)."""
    events = []
    append_event = events.append
    normalized = buffer.replace("\r\n", "\n")
    parts = normalized.split("\n\n")
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
            append_event({"data": event_data})
    return events, remaining


class StreamConverter:
    """Converts chat.completions SSE events to responses API SSE events.

    Maps Kimi API (OpenAI-compatible) stream to OpenAI Responses API stream:
    - Kimi delta.reasoning_content -> response.reasoning_text.delta
    - Kimi delta.content         -> response.output_text.delta
    """

    def __init__(self, response_id: str, model: str):
        self.response_id = response_id or f"resp_{uuid.uuid4().hex[:16]}"
        self.model = model
        self.created_at = int(time.time())
        self.item_id = f"msg_{self.response_id[-12:]}"
        self._reasoning_item_id = f"rs_{self.response_id[-12:]}"
        self._reasoning_output_index: int | None = None
        self._seq = 0
        self._preamble_sent = False
        self._in_reasoning = False
        self._reasoning_started = False
        self._reasoning_item_done = False
        self._reasoning_parts: list[str] = []
        self._text_parts: list[str] = []
        self._tool_calls: dict[int, dict] = {}
        self._emitted_tool_ids: set[int] = set()
        self._emitted_tool_call_items: set[int] = set()
        self._tool_call_output_indices: dict[int, int] = {}
        self._next_output_index = 1  # 0 is reserved for the message item
        self._completed = False
        self._message_done = False

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def _build_response_object(self, status: str, output: list[dict]) -> dict:
        """Build a Responses API response object compatible with SDK event parsing."""
        completed_at = self.created_at if status == "completed" else None
        return {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "completed_at": completed_at,
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": True,
            "previous_response_id": None,
            "reasoning": {
                "effort": None,
                "summary": None,
            },
            "temperature": 1,
            "text": {
                "format": {"type": "text"},
            },
            "tool_choice": "auto",
            "tools": [],
            "top_p": 1,
            "truncation": "disabled",
            "usage": None,
            "user": None,
            "metadata": {},
        }

    def get_preamble_events(self) -> list[str]:
        """Return the initial events required by OpenAI SDK."""
        if self._preamble_sent:
            return []
        self._preamble_sent = True
        return [
            json.dumps({
                "type": "response.created",
                "response": self._build_response_object(status="in_progress", output=[]),
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")),
            json.dumps({
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "in_progress",
                    "content": [],
                },
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")),
        ]

    def process_event(self, event_data: str) -> list[str]:
        """Process a single chat.completions SSE event.

        Returns a list of converted responses API event strings.
        """
        events: list[str] = []

        if event_data.strip() == "[DONE]":
            self._emit_completion_events(events)
            return events

        try:
            data = json.loads(event_data)
        except json.JSONDecodeError:
            return events

        choices = data.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        delta = choice.get("delta", {})
        content = delta.get("content")
        reasoning_content = delta.get("reasoning_content")
        tool_calls = delta.get("tool_calls")
        finish_reason = choice.get("finish_reason")

        # When upstream signals completion, emit completion events immediately.
        # Some APIs may not send [DONE] reliably, so we react to finish_reason.
        if finish_reason is not None and not self._completed:
            self._emit_completion_events(events)
            return events

        # Handle reasoning_content (Kimi API specific field)
        if reasoning_content is not None:
            if not self._in_reasoning and not self._reasoning_started:
                self._in_reasoning = True
                self._reasoning_started = True
                if self._reasoning_output_index is None:
                    self._reasoning_output_index = self._next_output_index
                    self._next_output_index += 1
                events.append(json.dumps({
                    "type": "response.output_item.added",
                    "output_index": self._reasoning_output_index,
                    "item": {
                        "id": self._reasoning_item_id,
                        "type": "reasoning",
                        "status": "in_progress",
                        "summary": [],
                        "content": [],
                    },
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                events.append(json.dumps({
                    "type": "response.content_part.added",
                    "item_id": self._reasoning_item_id,
                    "output_index": self._reasoning_output_index,
                    "content_index": 0,
                    "part": {"type": "reasoning_text", "text": ""},
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
            self._reasoning_parts.append(reasoning_content)
            events.append(json.dumps({
                "type": "response.reasoning_text.delta",
                "item_id": self._reasoning_item_id,
                "output_index": self._reasoning_output_index,
                "content_index": 0,
                "delta": reasoning_content,
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")))

        # Handle content
        if content is not None:
            # Transition from reasoning to content: end reasoning first
            if self._in_reasoning:
                self._in_reasoning = False
                events.append(json.dumps({
                    "type": "response.reasoning_text.done",
                    "item_id": self._reasoning_item_id,
                    "output_index": self._reasoning_output_index,
                    "content_index": 0,
                    "text": "".join(self._reasoning_parts),
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                if not self._reasoning_item_done:
                    self._reasoning_item_done = True
                    events.append(json.dumps({
                        "type": "response.output_item.done",
                        "output_index": self._reasoning_output_index,
                        "item": {
                            "id": self._reasoning_item_id,
                            "type": "reasoning",
                            "status": "completed",
                            "summary": [],
                            "content": [{
                                "type": "reasoning_text",
                                "text": "".join(self._reasoning_parts),
                            }],
                        },
                        "sequence_number": self._next_seq(),
                    }, separators=(",", ":")))

            # First actual content: emit content_part.added for output_text
            if content and not self._text_parts:
                events.append(json.dumps({
                    "type": "response.content_part.added",
                    "item_id": self.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

            self._text_parts.append(content)
            events.append(json.dumps({
                "type": "response.output_text.delta",
                "item_id": self.item_id,
                "output_index": 0,
                "content_index": 0,
                "delta": content,
                "logprobs": [],
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")))

        if tool_calls:
            reasoning_text = "".join(self._reasoning_parts)
            text_content = "".join(self._text_parts)
            # Transition from reasoning to tool_calls: close reasoning first
            if self._in_reasoning:
                self._in_reasoning = False
                events.append(json.dumps({
                    "type": "response.reasoning_text.done",
                    "item_id": self._reasoning_item_id,
                    "output_index": self._reasoning_output_index,
                    "content_index": 0,
                    "text": reasoning_text,
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                if not self._reasoning_item_done:
                    self._reasoning_item_done = True
                    events.append(json.dumps({
                        "type": "response.output_item.done",
                        "output_index": self._reasoning_output_index,
                        "item": {
                            "id": self._reasoning_item_id,
                            "type": "reasoning",
                            "status": "completed",
                            "summary": [],
                            "content": [{
                                "type": "reasoning_text",
                                "text": reasoning_text,
                            }],
                        },
                        "sequence_number": self._next_seq(),
                    }, separators=(",", ":")))
            # Transition from content to tool_calls: close content first
            if text_content and not self._message_done:
                events.append(json.dumps({
                    "type": "response.output_text.done",
                    "item_id": self.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text_content,
                    "logprobs": [],
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                events.append(json.dumps({
                    "type": "response.content_part.done",
                    "item_id": self.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text_content, "annotations": []},
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                self._message_done = True
            events.extend(self._process_tool_call_delta(tool_calls))

        return events

    def _emit_completion_events(self, events: list[str]) -> None:
        """Emit the final completion event sequence."""
        if self._completed:
            return
        self._completed = True
        reasoning_text = "".join(self._reasoning_parts)
        text_content = "".join(self._text_parts)

        # End reasoning if still in progress
        if self._in_reasoning:
            self._in_reasoning = False
            events.append(json.dumps({
                "type": "response.reasoning_text.done",
                "item_id": self._reasoning_item_id,
                "output_index": self._reasoning_output_index,
                "content_index": 0,
                "text": reasoning_text,
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")))
            if not self._reasoning_item_done:
                self._reasoning_item_done = True
                events.append(json.dumps({
                    "type": "response.output_item.done",
                    "output_index": self._reasoning_output_index,
                    "item": {
                        "id": self._reasoning_item_id,
                        "type": "reasoning",
                        "status": "completed",
                        "summary": [],
                        "content": [{
                            "type": "reasoning_text",
                            "text": reasoning_text,
                        }],
                    },
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

        # Message content parts for done events (reasoning lives in a separate output item)
        content_parts: list[dict] = []
        if text_content:
            content_parts.append({"type": "output_text", "text": text_content, "annotations": []})

        # Close message item if not yet done (covers: has content, no content+no tools, no content+has tools)
        if not self._message_done:
            if text_content:
                # output_text.done
                events.append(json.dumps({
                    "type": "response.output_text.done",
                    "item_id": self.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text_content,
                    "logprobs": [],
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

                # content_part.done for output_text
                events.append(json.dumps({
                    "type": "response.content_part.done",
                    "item_id": self.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text_content, "annotations": []},
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
            elif not self._tool_calls:
                # No content and no tool calls: still emit output_text.done (even if empty)
                events.append(json.dumps({
                    "type": "response.output_text.done",
                    "item_id": self.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text_content,
                    "logprobs": [],
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                events.append(json.dumps({
                    "type": "response.content_part.done",
                    "item_id": self.item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text_content, "annotations": []},
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

            # output_item.done for message (always emitted when closing)
            events.append(json.dumps({
                "type": "response.output_item.done",
                "output_index": 0,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": content_parts,
                },
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")))

        # Close any tool calls that were emitted
        for index, tc in self._tool_calls.items():
            if index in self._emitted_tool_ids and tc.get("item_id"):
                output_index = self._tool_call_output_indices.get(index, self._next_output_index)
                # function_call_arguments.done
                events.append(json.dumps({
                    "type": "response.function_call_arguments.done",
                    "item_id": tc["item_id"],
                    "output_index": output_index,
                    "call_id": tc.get("call_id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", ""),
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                # output_item.done for function_call
                events.append(json.dumps({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": {
                        "id": tc["item_id"],
                        "type": "function_call",
                        "status": "completed",
                        "call_id": tc.get("call_id", ""),
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", ""),
                    },
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

        # Build output array for response.completed
        output_items: list[dict] = []
        if reasoning_text:
            output_items.append({
                "id": self._reasoning_item_id,
                "type": "reasoning",
                "status": "completed",
                "summary": [],
                "content": [{
                    "type": "reasoning_text",
                    "text": reasoning_text,
                }],
            })
        if content_parts:
            output_items.append({
                "id": self.item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content_parts,
            })
        for index, tc in self._tool_calls.items():
            if tc.get("item_id"):
                output_items.append({
                    "id": tc["item_id"],
                    "type": "function_call",
                    "status": "completed",
                    "call_id": tc.get("call_id", ""),
                    "name": tc.get("name", ""),
                    "arguments": tc.get("arguments", ""),
                })

        # response.completed
        events.append(json.dumps({
            "type": "response.completed",
            "response": self._build_response_object(status="completed", output=output_items),
            "sequence_number": self._next_seq(),
        }, separators=(",", ":")))

    def _process_tool_call_delta(self, tool_calls: list[dict]) -> list[str]:
        """Accumulate tool call deltas and emit events.

        Returns a list of converted event strings (may include
        response.output_item.added + response.function_call_arguments.delta).
        """
        events: list[str] = []
        for tc in tool_calls:
            index = tc.get("index", 0)

            if index not in self._tool_calls:
                call_id = tc.get("id", "")
                item_id = f"fc_{call_id}" if call_id else ""
                self._tool_calls[index] = {
                    "call_id": call_id,
                    "item_id": item_id,
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", ""),
                }
            else:
                existing = self._tool_calls[index]
                if tc.get("id"):
                    existing["call_id"] = tc["id"]
                    existing["item_id"] = f"fc_{tc['id']}"
                func = tc.get("function", {})
                if func.get("name"):
                    existing["name"] = func["name"]
                if func.get("arguments"):
                    existing["arguments"] += func["arguments"]

            existing = self._tool_calls[index]

            if not existing.get("call_id") or not existing["name"] or not existing.get("item_id"):
                continue

            # Assign output_index for this tool call (persistent)
            if index not in self._tool_call_output_indices:
                self._tool_call_output_indices[index] = self._next_output_index
                self._next_output_index += 1
            output_index = self._tool_call_output_indices[index]

            # Emit output_item.added on first encounter
            if index not in self._emitted_tool_call_items:
                self._emitted_tool_call_items.add(index)
                events.append(json.dumps({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": {
                        "id": existing["item_id"],
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": existing["call_id"],
                        "name": existing["name"],
                        "arguments": "",
                    },
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

            # Track that we've emitted this tool call
            if index not in self._emitted_tool_ids:
                self._emitted_tool_ids.add(index)

            # Extract just the delta (new arguments fragment) from this chunk
            func = tc.get("function", {})
            arg_delta = func.get("arguments", "") or ""

            events.append(json.dumps({
                "type": "response.function_call_arguments.delta",
                "item_id": existing["item_id"],
                "output_index": output_index,
                "call_id": existing["call_id"],
                "delta": arg_delta,
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")))

        return events
