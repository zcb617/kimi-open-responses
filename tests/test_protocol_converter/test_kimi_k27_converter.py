import json

from src.kimi_k27_protocol_converter.request_converter import (
    _CUSTOM_TOOL_PROXY_PREFIX,
    _encode_tool_proxy,
    convert_request,
)
from src.kimi_k27_protocol_converter.stream_converter import StreamConverter


def _stream_custom_arguments(converter: StreamConverter, wrapper: str) -> tuple[list[str], int]:
    proxy_name = _encode_tool_proxy(_CUSTOM_TOOL_PROXY_PREFIX, "functions", "exec")
    deltas: list[str] = []
    first_delta_at = -1
    for index, fragment in enumerate(wrapper):
        function = {"arguments": fragment}
        tool_call = {"index": 0, "function": function}
        if index == 0:
            tool_call["id"] = "tool_exec"
            function["name"] = proxy_name
        events = converter.process_event(json.dumps({
            "choices": [{"delta": {"tool_calls": [tool_call]}, "finish_reason": None}],
        }))
        for event in events:
            parsed = json.loads(event)
            if parsed["type"] == "response.custom_tool_call_input.delta":
                if first_delta_at == -1:
                    first_delta_at = index
                deltas.append(parsed["delta"])
    return deltas, first_delta_at


def test_k27_request_uses_official_tool_and_thinking_defaults():
    converted = convert_request({
        "model": "kimi-k2.7-code",
        "input": "hello",
        "stream": True,
        "tool_choice": "required",
        "temperature": 0.2,
        "top_p": 0.5,
        "presence_penalty": 1,
        "frequency_penalty": 1,
    })

    assert converted["thinking"] == {"type": "enabled"}
    assert converted["tool_choice"] == "auto"
    assert "temperature" not in converted
    assert "top_p" not in converted
    assert "presence_penalty" not in converted
    assert "frequency_penalty" not in converted


def test_k27_custom_tool_input_streams_before_json_wrapper_is_complete():
    converter = StreamConverter(response_id="resp-k27-custom", model="kimi-k2.7-code")
    custom_input = 'const text = "你好😀";\nconsole.log(text);'
    wrapper = json.dumps({"input": custom_input})

    deltas, first_delta_at = _stream_custom_arguments(converter, wrapper)

    assert "".join(deltas) == custom_input
    assert 0 <= first_delta_at < len(wrapper) - 1


def test_k27_truncated_custom_tool_still_emits_available_input_prefix():
    converter = StreamConverter(response_id="resp-k27-truncated", model="kimi-k2.7-code")
    partial_wrapper = json.dumps({"input": "\nconst fs = require('fs');\nconst path = value"})[:-9]

    deltas, _ = _stream_custom_arguments(converter, partial_wrapper)

    assert "".join(deltas) == "\nconst fs = require('fs');\nconst path "
    assert converter._completed is False
