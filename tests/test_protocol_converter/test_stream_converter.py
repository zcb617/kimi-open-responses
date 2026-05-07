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


def test_empty_delta_skipped():
    converter = StreamConverter(response_id="resp-456", model="kimi-k2.6")

    # Events with no content should be skipped
    event = '{"id":"chatcmpl-1","choices":[{"delta":{"role":"assistant"}}]}'
    result = converter.process_event(event)
    assert result is None


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
