import json

from src.kimi_k3_protocol_converter.request_converter import (
    _NAMESPACE_TOOL_PROXY_PREFIX,
    _encode_tool_proxy,
    convert_request,
)
from src.kimi_k3_protocol_converter.stream_converter import StreamConverter


def _event_types(events: list[str]) -> list[str]:
    return [json.loads(event)["type"] for event in events]


def test_k3_stream_request_includes_final_usage():
    result = convert_request({"model": "kimi-k3", "input": "hello", "stream": True})

    assert result["stream_options"] == {"include_usage": True}


def test_k3_request_uses_k3_reasoning_and_tool_choice_fields():
    result = convert_request({
        "model": "kimi-k3",
        "input": "hello",
        "reasoning": {"effort": "high"},
        "tool_choice": "required",
        "temperature": 0.2,
        "top_p": 0.5,
        "presence_penalty": 1,
        "frequency_penalty": 1,
    })

    assert result["reasoning_effort"] == "high"
    assert result["tool_choice"] == "required"
    assert "thinking" not in result
    assert "temperature" not in result
    assert "top_p" not in result
    assert "presence_penalty" not in result
    assert "frequency_penalty" not in result


def test_k3_unsupported_reasoning_effort_uses_server_default():
    result = convert_request({
        "model": "kimi-k3",
        "input": "hello",
        "reasoning": {"effort": "medium"},
    })

    assert "reasoning_effort" not in result


def test_k3_waits_for_done_after_tool_finish_and_usage():
    converter = StreamConverter(response_id="resp-k3", model="kimi-k3")
    proxy_name = _encode_tool_proxy(
        _NAMESPACE_TOOL_PROXY_PREFIX,
        "mcp__fastctx",
        "glob",
    )
    converter.process_event(json.dumps({
        "choices": [{
            "delta": {
                "tool_calls": [{
                    "index": 0,
                    "id": "call_glob",
                    "function": {"name": proxy_name, "arguments": "{}"},
                }],
            },
            "finish_reason": None,
        }],
    }))

    finish_events = converter.process_event(json.dumps({
        "choices": [{"delta": {}, "finish_reason": "tool_calls"}],
    }))
    assert "response.completed" not in _event_types(finish_events)
    assert converter._completed is False

    usage_events = converter.process_event(json.dumps({
        "choices": [],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 80},
        },
    }))
    assert usage_events == []
    assert converter._completed is False

    done_events = converter.process_event("[DONE]")
    assert _event_types(done_events)[-1] == "response.completed"
    completed = json.loads(done_events[-1])
    assert completed["response"]["usage"]["total_tokens"] == 120
    function_call = next(
        item
        for item in completed["response"]["output"]
        if item["type"] == "function_call"
    )
    assert function_call["name"] == "glob"
    assert function_call["namespace"] == "mcp__fastctx"


def test_k3_done_is_idempotent():
    converter = StreamConverter(response_id="resp-k3-done", model="kimi-k3")

    first = converter.process_event("[DONE]")
    second = converter.process_event("[DONE]")

    assert _event_types(first)[-1] == "response.completed"
    assert second == []
