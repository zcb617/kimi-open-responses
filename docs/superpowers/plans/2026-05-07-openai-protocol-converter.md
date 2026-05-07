# OpenAI Protocol Converter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python library that converts OpenAI responses API requests/responses to chat.completions API format, targeting Kimi 2.6 upstream.

**Architecture:** Stateless pure-function converter with separate modules for request, response, and stream conversion. llm_router owns all state (conversation history, API key validation). TDD throughout.

**Tech Stack:** Python 3.10+, pytest, no external dependencies

---

## File Structure

```
src/openai_protocol_converter/
├── __init__.py              # Package exports
├── types.py                 # TypedDict definitions for both protocols
├── request_converter.py     # responses API → chat.completions
├── response_converter.py    # chat.completions → responses API
└── stream_converter.py      # SSE streaming conversion

tests/test_protocol_converter/
├── test_request_converter.py
├── test_response_converter.py
└── test_stream_converter.py
```

---

### Task 1: Project Skeleton

**Files:**
- Create: `src/openai_protocol_converter/__init__.py`
- Create: `src/openai_protocol_converter/types.py`
- Create: `tests/test_protocol_converter/__init__.py`

- [ ] **Step 1: Create directories**

```bash
mkdir -p src/openai_protocol_converter tests/test_protocol_converter
```

- [ ] **Step 2: Write types.py**

```python
"""Type definitions for OpenAI protocol converter."""
from typing import TypedDict, NotRequired


# --- chat.completions types ---

class ChatMessage(TypedDict):
    role: str
    content: str


class ChatToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: dict


class ChatTool(TypedDict):
    type: str
    function: ChatToolFunction


class ChatCompletionRequest(TypedDict):
    model: str
    messages: list[ChatMessage]
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]
    presence_penalty: NotRequired[float]
    frequency_penalty: NotRequired[float]
    tools: NotRequired[list[ChatTool]]
    tool_choice: NotRequired[str | dict]
    stream: NotRequired[bool]
    response_format: NotRequired[dict]
    thinking: NotRequired[dict]


class ChatCompletionChoice(TypedDict):
    index: int
    message: dict
    finish_reason: str | None


class ChatCompletionResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict


# --- responses API types ---

class ResponsesInputText(TypedDict):
    type: str
    text: str


class ResponsesInputMessage(TypedDict):
    role: str
    content: str | list[ResponsesInputText]


class ResponsesTextFormat(TypedDict):
    type: str
    json_schema: NotRequired[dict]


class ResponsesTextConfig(TypedDict):
    format: NotRequired[ResponsesTextFormat]


class ResponsesToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: NotRequired[dict]
    strict: NotRequired[bool]


class ResponsesTool(TypedDict):
    type: str
    function: ResponsesToolFunction


class ResponsesReasoning(TypedDict):
    effort: NotRequired[str]


class ResponsesRequest(TypedDict):
    model: str
    input: str | list[ResponsesInputMessage]
    instructions: NotRequired[str]
    temperature: NotRequired[float]
    max_output_tokens: NotRequired[int]
    top_p: NotRequired[float]
    presence_penalty: NotRequired[float]
    frequency_penalty: NotRequired[float]
    tools: NotRequired[list[ResponsesTool]]
    tool_choice: NotRequired[str | dict]
    stream: NotRequired[bool]
    text: NotRequired[ResponsesTextConfig]
    reasoning: NotRequired[ResponsesReasoning]
    previous_response_id: NotRequired[str]


class ResponsesOutputText(TypedDict):
    type: str
    text: str


class ResponsesOutputFunctionCall(TypedDict):
    type: str
    call_id: str
    name: str
    arguments: str


class ResponsesOutputItem(TypedDict):
    type: str
    role: NotRequired[str]
    content: NotRequired[list[ResponsesOutputText | ResponsesOutputFunctionCall]]


class ResponsesResponse(TypedDict):
    id: str
    object: str
    created_at: int
    model: str
    output: list[ResponsesOutputItem]
    usage: dict
    status: NotRequired[str]
```

- [ ] **Step 3: Write __init__.py**

```python
"""OpenAI Protocol Converter — responses API ↔ chat.completions for Kimi 2.6."""

from .request_converter import convert_request
from .response_converter import convert_response
from .stream_converter import StreamConverter

__all__ = ["convert_request", "convert_response", "StreamConverter"]
```

- [ ] **Step 4: Write tests __init__.py**

```bash
touch tests/test_protocol_converter/__init__.py
```

- [ ] **Step 5: Commit**

```bash
git add src/ tests/
git commit -m "feat: add project skeleton and type definitions

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: Request Converter — Core Fields

**Files:**
- Create: `src/openai_protocol_converter/request_converter.py`
- Create: `tests/test_protocol_converter/test_request_converter.py`

- [ ] **Step 1: Write failing test for input string conversion**

```python
import pytest
from src.openai_protocol_converter.request_converter import convert_request


def test_input_string_to_messages():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello, world!",
    }
    result = convert_request(req)
    assert result["messages"] == [{"role": "user", "content": "Hello, world!"}]
    assert result["model"] == "kimi-k2.6"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_request_converter.py::test_input_string_to_messages -v
```

Expected: `FAIL` — `ImportError` or `ModuleNotFoundError`

- [ ] **Step 3: Write minimal request_converter.py**

```python
"""Convert responses API requests to chat.completions format."""


def convert_request(responses_req: dict) -> dict:
    """Convert a responses API request dict to chat.completions format."""
    chat_req: dict = {"model": responses_req["model"]}

    # Handle input -> messages
    input_data = responses_req.get("input", "")
    if isinstance(input_data, str):
        chat_req["messages"] = [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        chat_req["messages"] = list(input_data)

    # Handle instructions -> system message
    instructions = responses_req.get("instructions")
    if instructions:
        chat_req["messages"].insert(0, {"role": "system", "content": instructions})

    # Parameter mapping
    if "temperature" in responses_req:
        chat_req["temperature"] = responses_req["temperature"]
    if "max_output_tokens" in responses_req:
        chat_req["max_tokens"] = responses_req["max_output_tokens"]
    if "top_p" in responses_req:
        chat_req["top_p"] = responses_req["top_p"]
    if "presence_penalty" in responses_req:
        chat_req["presence_penalty"] = responses_req["presence_penalty"]
    if "frequency_penalty" in responses_req:
        chat_req["frequency_penalty"] = responses_req["frequency_penalty"]
    if "tool_choice" in responses_req:
        chat_req["tool_choice"] = responses_req["tool_choice"]
    if "stream" in responses_req:
        chat_req["stream"] = responses_req["stream"]

    return chat_req
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_request_converter.py::test_input_string_to_messages -v
```

Expected: `PASS`

- [ ] **Step 5: Write test for input list passthrough**

```python
def test_input_list_passthrough():
    req = {
        "model": "kimi-k2.6",
        "input": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ],
    }
    result = convert_request(req)
    assert result["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello!"},
    ]
```

- [ ] **Step 6: Write test for instructions prepended as system message**

```python
def test_instructions_prepended_to_messages():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "instructions": "Be concise.",
    }
    result = convert_request(req)
    assert result["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Hello!"},
    ]
```

- [ ] **Step 7: Write test for parameter mapping**

```python
def test_parameter_mapping():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "temperature": 0.7,
        "max_output_tokens": 100,
        "top_p": 0.9,
        "presence_penalty": 0.5,
        "frequency_penalty": 0.3,
        "stream": True,
    }
    result = convert_request(req)
    assert result["temperature"] == 0.7
    assert result["max_tokens"] == 100
    assert result["top_p"] == 0.9
    assert result["presence_penalty"] == 0.5
    assert result["frequency_penalty"] == 0.3
    assert result["stream"] is True
```

- [ ] **Step 8: Run all request converter tests**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_request_converter.py -v
```

Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/openai_protocol_converter/request_converter.py tests/test_protocol_converter/test_request_converter.py
git commit -m "feat: add request converter core fields

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: Request Converter — Advanced Fields

**Files:**
- Modify: `src/openai_protocol_converter/request_converter.py`
- Modify: `tests/test_protocol_converter/test_request_converter.py`

- [ ] **Step 1: Write failing test for reasoning.effort mapping**

```python
def test_reasoning_effort_medium_to_enabled():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "reasoning": {"effort": "medium"},
    }
    result = convert_request(req)
    assert result["thinking"] == {"type": "enabled"}


def test_reasoning_effort_none_to_disabled():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "reasoning": {"effort": "none"},
    }
    result = convert_request(req)
    assert result["thinking"] == {"type": "disabled"}


def test_no_reasoning_no_thinking():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
    }
    result = convert_request(req)
    assert "thinking" not in result
```

- [ ] **Step 2: Write failing test for text.format → response_format**

```python
def test_text_format_json_schema():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "text": {
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "greeting",
                    "schema": {"type": "object"},
                },
            },
        },
    }
    result = convert_request(req)
    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "greeting",
            "schema": {"type": "object"},
        },
    }
```

- [ ] **Step 3: Write failing test for tools adaptation**

```python
def test_tools_adaptation():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {"type": "object"},
                },
            },
        ],
    }
    result = convert_request(req)
    assert result["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object"},
            },
        },
    ]
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_request_converter.py -v
```

Expected: New tests FAIL

- [ ] **Step 5: Extend request_converter.py**

Add to `convert_request()` after the parameter mapping block:

```python
    # Reasoning effort -> thinking.type
    reasoning = responses_req.get("reasoning")
    if reasoning:
        effort = reasoning.get("effort", "medium")
        if effort == "none":
            chat_req["thinking"] = {"type": "disabled"}
        else:
            chat_req["thinking"] = {"type": "enabled"}

    # text.format -> response_format
    text_config = responses_req.get("text")
    if text_config and "format" in text_config:
        fmt = text_config["format"]
        chat_req["response_format"] = dict(fmt)

    # Tools — pass through (already compatible)
    if "tools" in responses_req:
        chat_req["tools"] = responses_req["tools"]
```

- [ ] **Step 6: Run all tests**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_request_converter.py -v
```

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/openai_protocol_converter/request_converter.py tests/test_protocol_converter/test_request_converter.py
git commit -m "feat: add reasoning.effort, text.format, tools conversion

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: Response Converter — Core Fields

**Files:**
- Create: `src/openai_protocol_converter/response_converter.py`
- Create: `tests/test_protocol_converter/test_response_converter.py`

- [ ] **Step 1: Write failing test for basic response conversion**

```python
import pytest
from src.openai_protocol_converter.response_converter import convert_response


def test_basic_response_conversion():
    chat_resp = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "total_tokens": 17,
        },
    }
    result = convert_response(chat_resp)
    assert result["id"] == "chatcmpl-123"
    assert result["object"] == "response"
    assert result["created_at"] == 1700000000
    assert result["model"] == "kimi-k2.6"
    assert result["status"] == "completed"
    assert len(result["output"]) == 1
    assert result["output"][0]["type"] == "message"
    assert result["output"][0]["role"] == "assistant"
    assert len(result["output"][0]["content"]) == 1
    assert result["output"][0]["content"][0] == {
        "type": "output_text",
        "text": "Hello! How can I help?",
    }
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_response_converter.py::test_basic_response_conversion -v
```

Expected: FAIL

- [ ] **Step 3: Write minimal response_converter.py**

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_response_converter.py::test_basic_response_conversion -v
```

Expected: PASS

- [ ] **Step 5: Write test for empty content**

```python
def test_empty_content():
    chat_resp = {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1700000001,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }
    result = convert_response(chat_resp)
    assert result["output"][0]["content"][0]["text"] == ""
```

- [ ] **Step 6: Write test for usage mapping**

```python
def test_usage_mapping():
    chat_resp = {
        "id": "chatcmpl-789",
        "object": "chat.completion",
        "created": 1700000002,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }
    result = convert_response(chat_resp)
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 50
    assert result["usage"]["total_tokens"] == 150
```

- [ ] **Step 7: Run all response converter tests**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_response_converter.py -v
```

Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/openai_protocol_converter/response_converter.py tests/test_protocol_converter/test_response_converter.py
git commit -m "feat: add response converter core fields

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: Response Converter — Tool Calls and Refusal

**Files:**
- Modify: `src/openai_protocol_converter/response_converter.py`
- Modify: `tests/test_protocol_converter/test_response_converter.py`

- [ ] **Step 1: Write failing test for tool_calls**

```python
def test_tool_calls_conversion():
    chat_resp = {
        "id": "chatcmpl-tool",
        "object": "chat.completion",
        "created": 1700000003,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Beijing"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            },
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 15, "total_tokens": 35},
    }
    result = convert_response(chat_resp)
    output = result["output"][0]
    assert output["type"] == "message"
    assert output["role"] == "assistant"
    assert len(output["content"]) == 1
    assert output["content"][0] == {
        "type": "output_function_call",
        "call_id": "call_123",
        "name": "get_weather",
        "arguments": '{"city": "Beijing"}',
    }
```

- [ ] **Step 2: Write failing test for multiple tool_calls**

```python
def test_multiple_tool_calls():
    chat_resp = {
        "id": "chatcmpl-tools",
        "object": "chat.completion",
        "created": 1700000004,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "Beijing"}',
                            },
                        },
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {
                                "name": "get_time",
                                "arguments": '{"timezone": "UTC"}',
                            },
                        },
                    ],
                },
                "finish_reason": "tool_calls",
            },
        ],
        "usage": {"prompt_tokens": 25, "completion_tokens": 30, "total_tokens": 55},
    }
    result = convert_response(chat_resp)
    content = result["output"][0]["content"]
    assert len(content) == 2
    assert content[0]["type"] == "output_function_call"
    assert content[0]["call_id"] == "call_1"
    assert content[1]["call_id"] == "call_2"
```

- [ ] **Step 3: Write failing test for refusal**

```python
def test_refusal_conversion():
    chat_resp = {
        "id": "chatcmpl-refuse",
        "object": "chat.completion",
        "created": 1700000005,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "refusal": "I cannot help with that.",
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
    }
    result = convert_response(chat_resp)
    output = result["output"][0]
    assert output["type"] == "message"
    assert len(output["content"]) == 1
    assert output["content"][0] == {
        "type": "refusal",
        "refusal": "I cannot help with that.",
    }
```

- [ ] **Step 4: Run tests to verify they fail**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_response_converter.py -v
```

Expected: New tests FAIL

- [ ] **Step 5: Extend response_converter.py**

Replace the content building section in `convert_response()`:

```python
    # Build output content items
    content_items: list[dict] = []

    if message.get("refusal"):
        content_items.append({
            "type": "refusal",
            "refusal": message["refusal"],
        })
    elif message.get("tool_calls"):
        for tool_call in message["tool_calls"]:
            content_items.append({
                "type": "output_function_call",
                "call_id": tool_call["id"],
                "name": tool_call["function"]["name"],
                "arguments": tool_call["function"]["arguments"],
            })
    else:
        content = message.get("content", "") or ""
        content_items.append({"type": "output_text", "text": content})

    output_item: dict = {
        "type": "message",
        "role": message.get("role", "assistant"),
        "content": content_items,
    }
```

- [ ] **Step 6: Run all tests**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_response_converter.py -v
```

Expected: All PASS

- [ ] **Step 7: Commit**

```bash
git add src/openai_protocol_converter/response_converter.py tests/test_protocol_converter/test_response_converter.py
git commit -m "feat: add tool_calls and refusal conversion

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: Stream Converter — Text Streaming

**Files:**
- Create: `src/openai_protocol_converter/stream_converter.py`
- Create: `tests/test_protocol_converter/test_stream_converter.py`

- [ ] **Step 1: Write failing test for text stream conversion**

```python
import pytest
from src.openai_protocol_converter.stream_converter import StreamConverter


def test_text_stream_basic():
    converter = StreamConverter(response_id="resp-123", model="kimi-k2.6")

    # Simulate chat.completions SSE events
    event1 = '{"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}'
    event2 = '{"id":"chatcmpl-1","choices":[{"delta":{"content":" world"}}]}'
    done = "[DONE]"

    results = []
    results.append(converter.process_event(event1))
    results.append(converter.process_event(event2))
    results.append(converter.process_event(done))

    assert results[0] == '{"id":"resp-123","output":[{"type":"output_text","text":"Hello"}]}'
    assert results[1] == '{"id":"resp-123","output":[{"type":"output_text","text":" world"}]}'
    assert results[2] == '{"id":"resp-123","status":"completed"}'
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_stream_converter.py::test_text_stream_basic -v
```

Expected: FAIL

- [ ] **Step 3: Write minimal stream_converter.py**

```python
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
            return json.dumps({"id": self.response_id, "status": "completed"})

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
        })
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_stream_converter.py::test_text_stream_basic -v
```

Expected: PASS

- [ ] **Step 5: Write test for empty delta events**

```python
def test_empty_delta_skipped():
    converter = StreamConverter(response_id="resp-456", model="kimi-k2.6")

    # Events with no content should be skipped
    event = '{"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"}}]}'
    result = converter.process_event(event)
    assert result is None
```

- [ ] **Step 6: Write test for role-only events**

```python
def test_role_event_generates_output_item():
    converter = StreamConverter(response_id="resp-789", model="kimi-k2.6")

    # First event might have role but no content
    event1 = '{"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"}}]}'
    result1 = converter.process_event(event1)
    assert result1 is None  # Skip role-only events

    # Subsequent content events
    event2 = '{"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"}}]}'
    result2 = converter.process_event(event2)
    assert result2 == '{"id":"resp-789","output":[{"type":"output_text","text":"Hi"}]}'
```

- [ ] **Step 7: Run all stream converter tests**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_stream_converter.py -v
```

Expected: All PASS

- [ ] **Step 8: Commit**

```bash
git add src/openai_protocol_converter/stream_converter.py tests/test_protocol_converter/test_stream_converter.py
git commit -m "feat: add stream converter for text streaming

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 7: Stream Converter — Tool Call Streaming

**Files:**
- Modify: `src/openai_protocol_converter/stream_converter.py`
- Modify: `tests/test_protocol_converter/test_stream_converter.py`

- [ ] **Step 1: Write failing test for tool_call stream**

```python
def test_tool_call_stream():
    converter = StreamConverter(response_id="resp-tool", model="kimi-k2.6")

    # Tool call streaming: deltas arrive incrementally
    event1 = '{"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_"}}]}}]}'
    event2 = '{"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"ci"}}]}}]}'
    event3 = '{"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\\":\\"Beijing\\"}"}}]}}]}'
    done = "[DONE]"

    results = []
    results.append(converter.process_event(event1))
    results.append(converter.process_event(event2))
    results.append(converter.process_event(event3))
    results.append(converter.process_event(done))

    # Tool call events should accumulate and emit when complete
    # For simplicity, emit on each delta with accumulated state
    assert results[0] is None  # Incomplete, accumulate
    assert results[1] is None  # Still accumulating
    assert results[2] == '{"id":"resp-tool","output":[{"type":"output_function_call","call_id":"call_1","name":"get_","arguments":"{\\"city\\":\\"Beijing\\"}"}]}'
    assert results[3] == '{"id":"resp-tool","status":"completed"}'
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_stream_converter.py::test_tool_call_stream -v
```

Expected: FAIL

- [ ] **Step 3: Extend stream_converter.py for tool calls**

Replace the `StreamConverter` class:

```python
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
            return json.dumps({"id": self.response_id, "status": "completed"})

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
        if content:
            return json.dumps({
                "id": self.response_id,
                "output": [{"type": "output_text", "text": content}],
            })

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

            # Emit if this is the first time we've seen this tool call
            # and it has an ID (indicating the start is complete)
            if index not in self._emitted_tool_ids and existing["id"]:
                self._emitted_tool_ids.add(index)
                return json.dumps({
                    "id": self.response_id,
                    "output": [{
                        "type": "output_function_call",
                        "call_id": existing["id"],
                        "name": existing["name"],
                        "arguments": existing["arguments"],
                    }],
                })

            # Re-emit with updated arguments if we've already emitted this tool
            if index in self._emitted_tool_ids:
                return json.dumps({
                    "id": self.response_id,
                    "output": [{
                        "type": "output_function_call",
                        "call_id": existing["id"],
                        "name": existing["name"],
                        "arguments": existing["arguments"],
                    }],
                })

        return None
```

- [ ] **Step 4: Run all stream converter tests**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_stream_converter.py -v
```

Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/openai_protocol_converter/stream_converter.py tests/test_protocol_converter/test_stream_converter.py
git commit -m "feat: add tool call streaming support

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: Integration Test and Final Polish

**Files:**
- Create: `tests/test_protocol_converter/test_integration.py`
- Modify: `src/openai_protocol_converter/__init__.py`

- [ ] **Step 1: Write end-to-end integration test**

```python
"""End-to-end integration tests for the protocol converter."""
from src.openai_protocol_converter import convert_request, convert_response, StreamConverter


def test_full_text_request_response_cycle():
    # Client sends responses API request
    responses_req = {
        "model": "kimi-k2.6",
        "input": "What is the weather?",
        "temperature": 0.5,
        "max_output_tokens": 50,
    }

    # Convert to chat.completions
    chat_req = convert_request(responses_req)
    assert chat_req["model"] == "kimi-k2.6"
    assert chat_req["messages"] == [{"role": "user", "content": "What is the weather?"}]
    assert chat_req["temperature"] == 0.5
    assert chat_req["max_tokens"] == 50

    # Simulate upstream response (chat.completions format)
    chat_resp = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000100,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I don't have real-time weather data.",
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }

    # Convert back to responses API
    responses_resp = convert_response(chat_resp)
    assert responses_resp["id"] == "chatcmpl-test"
    assert responses_resp["object"] == "response"
    assert responses_resp["created_at"] == 1700000100
    assert responses_resp["output"][0]["content"][0]["text"] == "I don't have real-time weather data."
    assert responses_resp["usage"]["input_tokens"] == 5
    assert responses_resp["usage"]["output_tokens"] == 7


def test_streaming_end_to_end():
    converter = StreamConverter(response_id="resp-e2e", model="kimi-k2.6")

    events = [
        '{"choices":[{"delta":{"content":"Thinking"}}]}',
        '{"choices":[{"delta":{"content":"..."}}]}',
        '{"choices":[{"delta":{"content":" Done!"}}]}',
        "[DONE]",
    ]

    results = [converter.process_event(e) for e in events]
    assert results[0] == '{"id":"resp-e2e","output":[{"type":"output_text","text":"Thinking"}]}'
    assert results[1] == '{"id":"resp-e2e","output":[{"type":"output_text","text":"..."}]}'
    assert results[2] == '{"id":"resp-e2e","output":[{"type":"output_text","text":" Done!"}]}'
    assert results[3] == '{"id":"resp-e2e","status":"completed"}'
```

- [ ] **Step 2: Run integration tests**

```bash
cd /var/work/kimi-open-responses && pytest tests/test_protocol_converter/test_integration.py -v
```

Expected: All PASS

- [ ] **Step 3: Run full test suite**

```bash
cd /var/work/kimi-open-responses && pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_protocol_converter/test_integration.py
git commit -m "test: add end-to-end integration tests

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Self-Review

### Spec Coverage Check

| Spec Section | Plan Task |
|-------------|-----------|
| Request conversion (input→messages) | Task 2 |
| Request conversion (instructions, parameters) | Task 2 |
| Request conversion (reasoning.effort) | Task 3 |
| Request conversion (text.format) | Task 3 |
| Request conversion (tools) | Task 3 |
| Response conversion (choices→output) | Task 4 |
| Response conversion (tool_calls) | Task 5 |
| Response conversion (refusal) | Task 5 |
| Stream conversion (text) | Task 6 |
| Stream conversion (tool_calls) | Task 7 |
| previous_response_id (external handling) | Design doc only — llm_router integration |
| Error handling | Design doc — implemented via test assertions |

### Placeholder Scan

- ✅ No "TBD", "TODO", "implement later"
- ✅ No vague "add error handling" — specific behaviors tested
- ✅ All test code includes actual assertions
- ✅ No "similar to Task N" references

### Type Consistency

- ✅ `convert_request()` signature consistent across tasks
- ✅ `convert_response()` signature consistent
- ✅ `StreamConverter` constructor signature consistent
