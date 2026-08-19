"""Convert chat.completions SSE stream to responses API SSE format."""

import base64
import json
import time
import uuid


def _map_chat_usage_to_responses(chat_usage: dict | None) -> dict | None:
    """Map chat.completions usage shape to Responses usage shape."""
    if not isinstance(chat_usage, dict):
        return None
    prompt_tokens_details = chat_usage.get("prompt_tokens_details")
    completion_tokens_details = chat_usage.get("completion_tokens_details")
    cached_tokens = chat_usage.get("cached_tokens")
    if cached_tokens is None and isinstance(prompt_tokens_details, dict):
        cached_tokens = prompt_tokens_details.get("cached_tokens")
    reasoning_tokens = chat_usage.get("reasoning_tokens")
    if reasoning_tokens is None and isinstance(completion_tokens_details, dict):
        reasoning_tokens = completion_tokens_details.get("reasoning_tokens")
    return {
        "input_tokens": chat_usage.get("prompt_tokens", 0),
        "input_tokens_details": {
            "cached_tokens": cached_tokens or 0,
        },
        "output_tokens": chat_usage.get("completion_tokens", 0),
        "output_tokens_details": {
            "reasoning_tokens": reasoning_tokens or 0,
        },
        "total_tokens": chat_usage.get("total_tokens", 0),
    }


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


_CUSTOM_TOOL_PROXY_PREFIX = "__cf1_"
_NAMESPACE_TOOL_PROXY_PREFIX = "__nf1_"


def _decode_custom_tool_proxy(name: str) -> tuple[str, str] | None:
    """Decode the request converter's stable custom-tool proxy name."""
    if not isinstance(name, str) or not name.startswith(_CUSTOM_TOOL_PROXY_PREFIX):
        return None
    token = name[len(_CUSTOM_TOOL_PROXY_PREFIX) :]
    try:
        token += "=" * (-len(token) % 4)
        namespace, tool_name = json.loads(base64.urlsafe_b64decode(token).decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(namespace, str) or not isinstance(tool_name, str):
        return None
    return namespace, tool_name


def _decode_namespace_tool_proxy(name: str) -> tuple[str, str] | None:
    """Decode a namespaced function proxy name into (namespace, name)."""
    if not isinstance(name, str) or not name.startswith(_NAMESPACE_TOOL_PROXY_PREFIX):
        return None
    token = name[len(_NAMESPACE_TOOL_PROXY_PREFIX) :]
    try:
        token += "=" * (-len(token) % 4)
        namespace, tool_name = json.loads(base64.urlsafe_b64decode(token).decode("utf-8"))
    except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(namespace, str) or not isinstance(tool_name, str):
        return None
    return namespace, tool_name


def _custom_input_from_arguments(arguments: str) -> str | None:
    """Extract raw custom input from the Chat function JSON wrapper."""
    try:
        value = json.loads(arguments)
    except (TypeError, ValueError, json.JSONDecodeError):
        # Do not emit a delta until the complete JSON wrapper is parseable.
        return None
    if isinstance(value, dict) and isinstance(value.get("input"), str):
        return value["input"]
    return None


def _custom_input_prefix_from_arguments(arguments: str) -> str | None:
    """Decode the available prefix of a streamed ``{"input":"..."}`` wrapper."""
    if not isinstance(arguments, str):
        return None

    index = 0
    length = len(arguments)

    def skip_whitespace(position: int) -> int:
        while position < length and arguments[position] in " \t\r\n":
            position += 1
        return position

    index = skip_whitespace(index)
    if index >= length or arguments[index] != "{":
        return None
    index = skip_whitespace(index + 1)

    key = '"input"'
    available_key = arguments[index:index + len(key)]
    if not key.startswith(available_key) or len(available_key) != len(key):
        return None
    index = skip_whitespace(index + len(key))
    if index >= length or arguments[index] != ":":
        return None
    index = skip_whitespace(index + 1)
    if index >= length or arguments[index] != '"':
        return None
    index += 1

    decoded: list[str] = []
    simple_escapes = {
        '"': '"',
        "\\": "\\",
        "/": "/",
        "b": "\b",
        "f": "\f",
        "n": "\n",
        "r": "\r",
        "t": "\t",
    }
    while index < length:
        char = arguments[index]
        if char == '"':
            return "".join(decoded)
        if char != "\\":
            decoded.append(char)
            index += 1
            continue

        if index + 1 >= length:
            break
        escape = arguments[index + 1]
        if escape in simple_escapes:
            decoded.append(simple_escapes[escape])
            index += 2
            continue
        if escape != "u" or index + 6 > length:
            break
        try:
            codepoint = int(arguments[index + 2:index + 6], 16)
        except ValueError:
            return None

        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 12 > length or arguments[index + 6:index + 8] != "\\u":
                break
            try:
                low = int(arguments[index + 8:index + 12], 16)
            except ValueError:
                return None
            if not 0xDC00 <= low <= 0xDFFF:
                return None
            codepoint = 0x10000 + ((codepoint - 0xD800) << 10) + (low - 0xDC00)
            index += 12
        else:
            index += 6
        decoded.append(chr(codepoint))

    return "".join(decoded)


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
        self._message_output_index: int | None = None
        self._message_started = False
        self._next_output_index = 0
        self._completed = False
        self._message_done = False
        self._usage: dict | None = None

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
            "usage": self._usage,
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
                "type": "response.in_progress",
                "response": self._build_response_object(status="in_progress", output=[]),
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")),
        ]

    def _ensure_message_started(self, events: list[str]) -> None:
        if self._message_started:
            return

        self._message_started = True
        self._message_output_index = self._next_output_index
        self._next_output_index += 1
        events.append(json.dumps({
            "type": "response.output_item.added",
            "output_index": self._message_output_index,
            "item": {
                "id": self.item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
            "sequence_number": self._next_seq(),
        }, separators=(",", ":")))

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

        mapped_usage = _map_chat_usage_to_responses(data.get("usage"))
        if mapped_usage is not None:
            self._usage = mapped_usage

        choices = data.get("choices", [])
        if not choices:
            return events

        choice = choices[0]
        mapped_usage = _map_chat_usage_to_responses(choice.get("usage"))
        if mapped_usage is not None:
            self._usage = mapped_usage
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
        if content:
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
            if not self._text_parts:
                self._ensure_message_started(events)
                events.append(json.dumps({
                    "type": "response.content_part.added",
                    "item_id": self.item_id,
                    "output_index": self._message_output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

            self._text_parts.append(content)
            events.append(json.dumps({
                "type": "response.output_text.delta",
                "item_id": self.item_id,
                "output_index": self._message_output_index,
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
                    "output_index": self._message_output_index,
                    "content_index": 0,
                    "text": text_content,
                    "logprobs": [],
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                events.append(json.dumps({
                    "type": "response.content_part.done",
                    "item_id": self.item_id,
                    "output_index": self._message_output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text_content, "annotations": []},
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
                events.append(json.dumps({
                    "type": "response.output_item.done",
                    "output_index": self._message_output_index,
                    "item": {
                        "id": self.item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{
                            "type": "output_text",
                            "text": text_content,
                            "annotations": [],
                        }],
                    },
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

        # Close the message only if a message item was actually started.
        if self._message_started and not self._message_done:
            if text_content:
                # output_text.done
                events.append(json.dumps({
                    "type": "response.output_text.done",
                    "item_id": self.item_id,
                    "output_index": self._message_output_index,
                    "content_index": 0,
                    "text": text_content,
                    "logprobs": [],
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

                # content_part.done for output_text
                events.append(json.dumps({
                    "type": "response.content_part.done",
                    "item_id": self.item_id,
                    "output_index": self._message_output_index,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text_content, "annotations": []},
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))
            # output_item.done for the started message
            events.append(json.dumps({
                "type": "response.output_item.done",
                "output_index": self._message_output_index,
                "item": {
                    "id": self.item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": content_parts,
                },
                "sequence_number": self._next_seq(),
            }, separators=(",", ":")))
            self._message_done = True

        # Close any tool calls that were emitted
        for index, tc in self._tool_calls.items():
            if index in self._emitted_tool_ids and tc.get("item_id"):
                output_index = self._tool_call_output_indices.get(index, self._next_output_index)
                if tc.get("custom"):
                    # Responses has dedicated custom-tool input events; do not
                    # expose the proxy as an ordinary function call.
                    custom_input = tc.get("custom_input")
                    if custom_input is None:
                        custom_input = _custom_input_from_arguments(tc.get("arguments", "")) or ""
                    events.append(json.dumps({
                        "type": "response.custom_tool_call_input.done",
                        "item_id": tc["item_id"],
                        "output_index": output_index,
                        "input": custom_input,
                        "sequence_number": self._next_seq(),
                    }, separators=(",", ":")))
                    done_item = {
                        "id": tc["item_id"],
                        "type": "custom_tool_call",
                        "status": "completed",
                        "call_id": tc.get("call_id", ""),
                        "name": tc.get("name", ""),
                        "input": custom_input,
                    }
                else:
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
                    done_item = {
                        "id": tc["item_id"],
                        "type": "function_call",
                        "status": "completed",
                        "call_id": tc.get("call_id", ""),
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", ""),
                    }
                if tc.get("namespace"):
                    done_item["namespace"] = tc["namespace"]
                # output_item.done closes either a function or custom call.
                events.append(json.dumps({
                    "type": "response.output_item.done",
                    "output_index": output_index,
                    "item": done_item,
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
                output_item = {
                    "id": tc["item_id"],
                    "type": "custom_tool_call" if tc.get("custom") else "function_call",
                    "status": "completed",
                    "call_id": tc.get("call_id", ""),
                    "name": tc.get("name", ""),
                    "input" if tc.get("custom") else "arguments": (
                        tc.get("custom_input")
                        if tc.get("custom_input") is not None
                        else (_custom_input_from_arguments(tc.get("arguments", "")) or "")
                        if tc.get("custom") else tc.get("arguments", "")
                    ),
                }
                if tc.get("namespace"):
                    output_item["namespace"] = tc["namespace"]
                output_items.append(output_item)

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
                proxy_name = tc.get("function", {}).get("name", "")
                custom_proxy = _decode_custom_tool_proxy(proxy_name)
                namespace_proxy = _decode_namespace_tool_proxy(proxy_name)
                self._tool_calls[index] = {
                    "call_id": call_id,
                    "item_id": item_id,
                    "name": (custom_proxy or namespace_proxy or ("", proxy_name))[1],
                    "namespace": namespace_proxy[0] if namespace_proxy and not custom_proxy else None,
                    "custom": custom_proxy is not None,
                    "arguments": tc.get("function", {}).get("arguments", ""),
                    "custom_input": "",
                }
            else:
                existing = self._tool_calls[index]
                if tc.get("id"):
                    existing["call_id"] = tc["id"]
                    existing["item_id"] = f"fc_{tc['id']}"
                func = tc.get("function", {})
                if func.get("name"):
                    custom_proxy = _decode_custom_tool_proxy(func["name"])
                    namespace_proxy = _decode_namespace_tool_proxy(func["name"])
                    existing["name"] = (custom_proxy or namespace_proxy or ("", func["name"]))[1]
                    existing["namespace"] = namespace_proxy[0] if namespace_proxy and not custom_proxy else None
                    existing["custom"] = custom_proxy is not None
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
                if existing.get("custom"):
                    added_item = {
                        "id": existing["item_id"],
                        "type": "custom_tool_call",
                        "status": "in_progress",
                        "call_id": existing["call_id"],
                        "name": existing["name"],
                        "input": "",
                    }
                else:
                    added_item = {
                        "id": existing["item_id"],
                        "type": "function_call",
                        "status": "in_progress",
                        "call_id": existing["call_id"],
                        "name": existing["name"],
                        "arguments": "",
                    }
                if existing.get("namespace"):
                    added_item["namespace"] = existing["namespace"]
                events.append(json.dumps({
                    "type": "response.output_item.added",
                    "output_index": output_index,
                    "item": added_item,
                    "sequence_number": self._next_seq(),
                }, separators=(",", ":")))

            # Track that we've emitted this tool call
            if index not in self._emitted_tool_ids:
                self._emitted_tool_ids.add(index)

            # Extract just the delta (new arguments fragment) from this chunk
            func = tc.get("function", {})
            arg_delta = func.get("arguments", "") or ""

            if existing.get("custom"):
                full_input = _custom_input_prefix_from_arguments(existing.get("arguments", ""))
                if full_input is None:
                    continue
                previous_input = existing.get("custom_input", "")
                input_delta = full_input[len(previous_input):] if full_input.startswith(previous_input) else full_input
                existing["custom_input"] = full_input
                if not input_delta:
                    continue
                delta_event = {
                    "type": "response.custom_tool_call_input.delta",
                    "item_id": existing["item_id"],
                    "output_index": output_index,
                    "delta": input_delta,
                    "sequence_number": self._next_seq(),
                }
            else:
                delta_event = {
                    "type": "response.function_call_arguments.delta",
                    "item_id": existing["item_id"],
                    "output_index": output_index,
                    "call_id": existing["call_id"],
                    "delta": arg_delta,
                    "sequence_number": self._next_seq(),
                }
            events.append(json.dumps(delta_event, separators=(",", ":")))

        return events
