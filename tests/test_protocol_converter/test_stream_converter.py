import json

from src.openai_protocol_converter.stream_converter import StreamConverter


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
