# OpenAI Protocol Converter Design

## 1. Overview

A Python library that converts OpenAI **responses API** (new protocol) requests/responses to **chat.completions API** (old protocol) format, specifically targeting **Kimi 2.6** as the upstream provider.

**Conversion direction:** responses API → chat.completions (request), chat.completions → responses API (response)

**Integration target:** [llm_router](https://github.com/...) project — a mitmproxy-based LLM routing proxy.

## 2. Architecture

```
Client (responses API)
    │
    ▼
┌─────────────────────────────────────────────────┐
│ llm_router (mitmproxy)                          │
│                                                 │
│  ┌─────────────────────────────────────────┐    │
│  │ openai_protocol_converter               │    │
│  │                                         │    │
│  │  ┌─────────────┐    ┌─────────────┐    │    │
│  │  │ request_    │───▶│ chat.compl. │    │    │
│  │  │ converter   │    │ request     │    │    │
│  │  └─────────────┘    └─────────────┘    │    │
│  │                                         │    │
│  │  ┌─────────────┐    ┌─────────────┐    │    │
│  │  │ response_   │◀───│ chat.compl. │    │    │
│  │  │ converter   │    │ response    │    │    │
│  │  └─────────────┘    └─────────────┘    │    │
│  │                                         │    │
│  │  ┌─────────────┐                       │    │
│  │  │ stream_     │───▶ SSE conversion   │    │
│  │  │ converter   │                       │    │
│  │  └─────────────┘                       │    │
│  └─────────────────────────────────────────┘    │
└─────────────────────────────────────────────────┘
    │
    ▼
Upstream (Kimi 2.6, chat.completions)
```

The converter is a **stateless, pure function library** — no I/O, no network calls. `llm_router` owns all state (API key validation, conversation history storage, response ID tracking).

## 3. Module Design

```
src/openai_protocol_converter/
├── __init__.py              # Exports: convert_request, convert_response, convert_stream_chunk
├── types.py                 # Shared TypedDict / dataclass definitions
├── request_converter.py     # responses API → chat.completions request conversion
├── response_converter.py    # chat.completions → responses API response conversion
└── stream_converter.py      # SSE streaming event conversion
```

| Module | Responsibility |
|--------|---------------|
| `request_converter` | Input field mapping: `input`→`messages`, `instructions`→system message, parameter renaming, tool format adaptation |
| `response_converter` | Response body reconstruction: `choices`→`output` array, usage field mapping, output item type detection |
| `stream_converter` | Per-event SSE conversion: delta accumulation, output item state tracking, completion signaling |
| `types.py` | Shared type definitions for both protocol formats |

## 4. Data Flow

### 4.1 Request Conversion

| responses API Field | chat.completions Field | Conversion Logic |
|---------------------|----------------------|------------------|
| `input` (string) | `messages` | `[{"role": "user", "content": input}]` |
| `input` (list) | `messages` | Pass through directly (already messages format) |
| `instructions` | `messages` | Prepend as `{"role": "system", "content": instructions}` |
| `model` | `model` | Pass through |
| `temperature` | `temperature` | Pass through |
| `max_output_tokens` | `max_tokens` | Rename |
| `top_p` | `top_p` | Pass through |
| `presence_penalty` | `presence_penalty` | Pass through |
| `frequency_penalty` | `frequency_penalty` | Pass through |
| `tools` | `tools` | Adapt format (wrap in `type: "function"`) |
| `tool_choice` | `tool_choice` | Pass through |
| `stream` | `stream` | Pass through |
| `text.format` (JSON Schema) | `response_format` | `{"type": "json_schema", "json_schema": {...}}` |
| `reasoning.effort` | `thinking.type` | `"none"` → `"disabled"`, others → `"enabled"` |
| `previous_response_id` | `messages` | **Not handled by converter** — injected by llm_router before conversion |

### 4.2 Response Conversion

| chat.completions Field | responses API Field | Conversion Logic |
|----------------------|---------------------|------------------|
| `choices[0].message.content` | `output[0].content` | Wrap as `{"type": "output_text", "text": content}` |
| `choices[0].message.tool_calls` | `output[0].content` | Each tool_call → `{"type": "output_function_call", ...}` |
| `choices[0].message.role` | `output[0].role` | Pass through |
| `choices[0].message.refusal` | `output[0]` | `{"type": "refusal", "refusal": ...}` |
| `usage` | `usage` | Map field names (`completion_tokens` → `output_tokens`, etc.) |
| `id` | `id` | Pass through |
| `created` | `created_at` | Pass through (Unix timestamp) |

### 4.3 Stream Conversion

**chat.completions SSE format:**
```
data: {"id":"...","choices":[{"delta":{"content":"Hello"}}]}
data: {"id":"...","choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

**responses API SSE format:**
```
data: {"id":"...","output":[{"type":"output_text","text":"Hello"}]}
data: {"id":"...","output":[{"type":"output_text","text":" world"}]}
data: {"id":"...","status":"completed"}
```

**Strategy:**
- `stream_converter` maintains an internal buffer to parse chat.completions `delta` events
- Accumulates tokens and emits corresponding responses API `output` events
- On `[DONE]`, emits `{"status": "completed"}`
- For `tool_calls` deltas, accumulates complete function call arguments before emitting `output_function_call`

## 5. `previous_response_id` Handling

Since the converter is stateless, conversation history loading happens in `llm_router`:

1. `llm_router` receives a request with `previous_response_id`
2. Queries database: `SELECT request_body, response_body FROM llm_calls WHERE call_id = ? AND api_key_id = ?`
3. If not found → return `invalid_request_error` with code `invalid_id`
4. If found but belongs to a different API key → same error (information hiding)
5. Extracts historical messages from the previous call's input/output
6. Injects them into the current request's `messages` array before calling the converter

**Isolation:** API key-level. Shared API key = shared conversation history (consistent with OpenAI's behavior).

## 6. `reasoning.effort` Handling (Kimi-specific)

| `reasoning.effort` Value | Kimi `thinking.type` |
|-------------------------|---------------------|
| `"none"` | `"disabled"` |
| `"low"` / `"medium"` / `"high"` | `"enabled"` |
| Not provided (defaults to `"medium"`) | `"enabled"` (Kimi's default) |

If `reasoning` object is not present in the request, no `thinking` field is set in the converted request, letting Kimi use its default behavior.

## 7. Error Handling

| Scenario | Handling |
|----------|----------|
| `previous_response_id` not found | Return `{"error": {"type": "invalid_request_error", "code": "invalid_id"}}` |
| `previous_response_id` belongs to different API key | Same as above (no information leak) |
| Unsupported field in old protocol | Log warning, drop field, continue processing |
| Invalid request body format | Return 400 with `invalid_request_error` |
| Upstream returns non-200 | Pass through upstream status code and error body |
| Stream conversion upstream disconnect | Emit `{"status": "incomplete"}`, close SSE |

## 8. Testing Strategy

```
tests/test_protocol_converter/
├── test_request_converter.py
├── test_response_converter.py
├── test_stream_converter.py
└── test_integration.py
```

Core test coverage:
- `input` as string → messages conversion
- `input` as list → pass-through
- `instructions` → system message prepending
- `text.format` (JSON Schema) → `response_format` mapping
- `reasoning.effort` → `thinking.type` mapping (all values)
- `previous_response_id` context injection (mocked external storage)
- Non-streaming response `choices` → `output` array
- Streaming SSE per-token conversion
- Tool call response → `output_function_call` events

## 9. Integration with llm_router

### 9.1 Database Schema Addition

```sql
-- Add protocol_version to model_configs table
ALTER TABLE model_configs ADD COLUMN protocol_version TEXT DEFAULT 'chat_completions';
-- Values: 'chat_completions' | 'responses_api'
```

### 9.2 Proxy Flow Changes

In `proxy.py`:

**Request path:**
```python
# After extracting model and matching mapping
if mapping.get("protocol_version") == "responses_api":
    # 1. Handle previous_response_id if present
    if previous_id := extract_previous_response_id(body):
        history = storage.get_call_history(previous_id, api_key_id)
        body = inject_history(body, history)

    # 2. Convert request
    chat_body = request_converter.convert_request(body)
    flow.request.content = chat_body.encode()
```

**Response path:**
```python
# After receiving upstream response
if mapping.get("protocol_version") == "responses_api":
    if is_stream:
        # Stream conversion handled by stream_converter
        flow.response.stream = create_stream_converter()
    else:
        resp_body = response_converter.convert_response(upstream_body)
        flow.response.content = resp_body.encode()
```

### 9.3 Response ID Storage

When saving to `llm_calls` table, ensure `call_id` (which serves as `response_id`) is a UUID generated before conversion, so it can be referenced by subsequent `previous_response_id` requests.

## 10. Design Decisions

1. **Stateless converter:** The converter itself has no state. All conversation history, API key validation, and response ID tracking is owned by `llm_router`. This keeps the converter testable and reusable.

2. **API key-level isolation:** `previous_response_id` lookups are scoped to the requesting API key. This matches OpenAI's behavior where shared API keys share conversation history.

3. **Kimi-first, extensible:** The converter targets Kimi 2.6 specifically for the initial implementation (especially `thinking.type` mapping). The architecture supports adding other upstream providers via a `provider` parameter in the future.

4. **Field dropping with warnings:** Unsupported fields (e.g., responses API features that have no chat.completions equivalent) are dropped with a warning log rather than failing the request. This maximizes compatibility.
