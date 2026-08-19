import json

from src.openai_protocol_converter.stream_converter import StreamConverter
from src.openai_protocol_converter.request_converter import (
    _CUSTOM_TOOL_PROXY_PREFIX,
    _NAMESPACE_TOOL_PROXY_PREFIX,
    _encode_tool_proxy,
)


def _parse_events(results):
    """Flatten list of event-string lists and parse JSON."""
    events = []
    for r in results:
        if r is None:
            continue
        if isinstance(r, list):
            for item in r:
                events.append(json.loads(item))
        else:
            events.append(json.loads(r))
    return events


def test_text_stream_basic():
    converter = StreamConverter(response_id="resp-123", model="kimi-k2.6")

    event1 = '{"id":"chatcmpl-1","choices":[{"delta":{"content":"Hello"}}]}'
    event2 = '{"id":"chatcmpl-1","choices":[{"delta":{"content":" world"}}]}'
    done = "[DONE]"

    results = []
    results.extend(converter.process_event(event1))
    results.extend(converter.process_event(event2))
    results.extend(converter.process_event(done))

    events = _parse_events(results)
    types = [e["type"] for e in events]

    assert types == [
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]

    assert events[1]["delta"] == "Hello"
    assert events[2]["delta"] == " world"
    assert events[3]["text"] == "Hello world"


def test_empty_delta_skipped():
    converter = StreamConverter(response_id="resp-456", model="kimi-k2.6")

    event = '{"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"}}]}'
    result = converter.process_event(event)
    assert result == []


def test_role_event_generates_output_item():
    converter = StreamConverter(response_id="resp-789", model="kimi-k2.6")

    event1 = '{"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"}}]}'
    result1 = converter.process_event(event1)
    assert result1 == []

    event2 = '{"id":"chatcmpl-1","choices":[{"delta":{"content":"Hi"}}]}'
    result2 = converter.process_event(event2)
    events = _parse_events(result2)
    assert events[0]["type"] == "response.content_part.added"
    assert events[1]["type"] == "response.output_text.delta"
    assert events[1]["delta"] == "Hi"


def test_empty_string_content_emitted():
    """Empty string content should emit output_text.delta, not skip."""
    converter = StreamConverter(response_id="resp-empty", model="kimi-k2.6")

    event = '{"id":"chatcmpl-1","choices":[{"delta":{"content":""}}]}'
    result = converter.process_event(event)
    events = _parse_events(result)
    assert len(events) == 1
    assert events[0]["type"] == "response.output_text.delta"
    assert events[0]["delta"] == ""


def test_tool_call_stream():
    converter = StreamConverter(response_id="resp-tool", model="kimi-k2.6")

    event1 = '{"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_"}}]}}]}'
    event2 = '{"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"ci"}}]}}]}'
    event3 = '{"id":"chatcmpl-1","choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"ty\\":\\"Beijing\\"}"}}]}}]}'
    done = "[DONE]"

    results = []
    results.extend(converter.process_event(event1))
    results.extend(converter.process_event(event2))
    results.extend(converter.process_event(event3))
    results.extend(converter.process_event(done))

    events = _parse_events(results)
    types = [e["type"] for e in events]

    assert types == [
        "response.output_item.added",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.delta",
        "response.output_item.done",
        "response.function_call_arguments.done",
        "response.output_item.done",
        "response.completed",
    ]

    # output_item.added for function_call
    added = events[0]
    assert added["item"]["type"] == "function_call"
    assert added["item"]["call_id"] == "call_1"
    assert added["item"]["name"] == "get_"

    # arguments accumulate (first event has no arguments field → empty delta)
    assert events[1]["delta"] == ""
    assert events[2]["delta"] == '{"ci'
    assert events[3]["delta"] == 'ty":"Beijing"}'

    # final arguments.done
    done_event = events[5]
    assert done_event["arguments"] == '{"city":"Beijing"}'


def test_custom_tool_stream_uses_custom_responses_events():
    converter = StreamConverter(response_id="resp-custom", model="kimi-k3")
    proxy_name = _encode_tool_proxy(_CUSTOM_TOOL_PROXY_PREFIX, "shell", "exec")
    first = converter.process_event(json.dumps({"choices": [{"delta": {
        "tool_calls": [{"index": 0, "id": "call_exec", "function": {
            "name": proxy_name, "arguments": "{\"input\":\"ls"}}]
    }}]}))
    added = json.loads(first[0])
    assert added["item"]["type"] == "custom_tool_call"
    assert added["item"]["name"] == "exec"
    assert "namespace" not in added["item"]
    assert len(first) == 1

    delta = converter.process_event(json.dumps({"choices": [{"delta": {
        "tool_calls": [{"index": 0, "function": {"arguments": " -la\"}"}}]
    }}]}))
    assert json.loads(delta[0])["type"] == "response.custom_tool_call_input.delta"

    done = converter.process_event("[DONE]")
    event_types = [json.loads(event)["type"] for event in done]
    assert "response.custom_tool_call_input.done" in event_types
    custom_done = next(
        json.loads(event)["item"]
        for event in done
        if json.loads(event)["type"] == "response.output_item.done"
        and json.loads(event)["item"].get("type") == "custom_tool_call"
    )
    assert "namespace" not in custom_done
    completed = json.loads(done[-1])
    custom_item = next(item for item in completed["response"]["output"] if item["type"] == "custom_tool_call")
    assert custom_item["input"] == "ls -la"
    assert "namespace" not in custom_item


def test_namespaced_function_stream_restores_namespace_on_all_items():
    converter = StreamConverter(response_id="resp-namespace", model="kimi-k3")
    proxy_name = _encode_tool_proxy(_NAMESPACE_TOOL_PROXY_PREFIX, "collaboration", "interrupt_agent")
    first = converter.process_event(json.dumps({"choices": [{"delta": {
        "tool_calls": [{"index": 0, "id": "call_interrupt", "function": {
            "name": proxy_name, "arguments": "{}"}}]
    }}]}))
    added = json.loads(first[0])
    assert added["item"]["type"] == "function_call"
    assert added["item"]["name"] == "interrupt_agent"
    assert added["item"]["namespace"] == "collaboration"

    done = converter.process_event("[DONE]")
    done_item = next(
        json.loads(event)["item"]
        for event in done
        if json.loads(event)["type"] == "response.output_item.done"
        and json.loads(event)["item"].get("type") == "function_call"
    )
    assert done_item["namespace"] == "collaboration"
    completed = json.loads(done[-1])
    output_item = next(item for item in completed["response"]["output"] if item["type"] == "function_call")
    assert output_item["namespace"] == "collaboration"


def test_custom_tool_input_split_before_key_does_not_emit_wrapper_delta():
    converter = StreamConverter(response_id="resp-custom-split", model="kimi-k3")
    proxy_name = _encode_tool_proxy(_CUSTOM_TOOL_PROXY_PREFIX, "shell", "exec")
    first = converter.process_event(json.dumps({"choices": [{"delta": {
        "tool_calls": [{"index": 0, "id": "call_split", "function": {
            "name": proxy_name, "arguments": "{\"inp"}}]
    }}]}))
    assert len(first) == 1

    second = converter.process_event(json.dumps({"choices": [{"delta": {
        "tool_calls": [{"index": 0, "function": {"arguments": "ut\":\"ok\"}"}}]
    }}]}))
    assert json.loads(second[0])["delta"] == "ok"
    completed = converter.process_event("[DONE]")
    final = json.loads(completed[-1])
    custom_item = next(item for item in final["response"]["output"] if item["type"] == "custom_tool_call")
    assert custom_item["input"] == "ok"


def test_custom_tool_input_escaped_characters_are_emitted_once():
    converter = StreamConverter(response_id="resp-custom-escaped", model="kimi-k3")
    proxy_name = _encode_tool_proxy(_CUSTOM_TOOL_PROXY_PREFIX, "shell", "exec")
    raw_input = 'a"b\\c\nd'
    arguments = json.dumps({"input": raw_input}, ensure_ascii=False)
    events = converter.process_event(json.dumps({"choices": [{"delta": {
        "tool_calls": [{"index": 0, "id": "call_escaped", "function": {
            "name": proxy_name, "arguments": arguments}}]
    }}]}))
    deltas = [json.loads(event)["delta"] for event in events if json.loads(event)["type"] == "response.custom_tool_call_input.delta"]
    assert deltas == [raw_input]
    completed = converter.process_event("[DONE]")
    final = json.loads(completed[-1])
    custom_item = next(item for item in final["response"]["output"] if item["type"] == "custom_tool_call")
    assert custom_item["input"] == raw_input


def test_finish_chunk_usage_mapped_into_response_completed():
    converter = StreamConverter(response_id="resp-usage", model="kimi-k2.6")

    finish_with_usage = json.dumps({
        "id": "chatcmpl-usage",
        "choices": [{
            "delta": {},
            "finish_reason": "stop",
            "usage": {
                "prompt_tokens": 123,
                "completion_tokens": 45,
                "total_tokens": 168,
                "cached_tokens": 100,
            },
        }],
    })
    results = converter.process_event(finish_with_usage)
    events = _parse_events(results)
    assert events[-1]["type"] == "response.completed"
    usage = events[-1]["response"]["usage"]
    assert usage["input_tokens"] == 123
    assert usage["output_tokens"] == 45
    assert usage["total_tokens"] == 168
    assert usage["input_tokens_details"]["cached_tokens"] == 100


def test_usage_only_chunk_updates_converter_usage():
    converter = StreamConverter(response_id="resp-usage-only", model="kimi-k2.6")

    # Completion may be emitted before a later usage-only chunk.
    converter.process_event('{"id":"chatcmpl-1","choices":[{"delta":{},"finish_reason":"stop"}]}')
    result = converter.process_event(json.dumps({
        "id": "chatcmpl-1",
        "choices": [],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
            "prompt_tokens_details": {"cached_tokens": 7},
        },
    }))
    assert result == []
    assert converter._usage["input_tokens"] == 10
    assert converter._usage["output_tokens"] == 3
    assert converter._usage["input_tokens_details"]["cached_tokens"] == 7
