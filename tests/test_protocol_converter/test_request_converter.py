from src.openai_protocol_converter.request_converter import convert_request


def test_input_string_to_messages():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello, world!",
    }
    result = convert_request(req)
    assert result["messages"] == [{"role": "user", "content": "Hello, world!"}]
    assert result["model"] == "kimi-k2.6"


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
    assert result["max_completion_tokens"] == 100
    assert result["top_p"] == 0.9
    assert result["presence_penalty"] == 0.5
    assert result["frequency_penalty"] == 0.3
    assert result["stream"] is True


def test_reasoning_effort_medium_to_enabled():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "reasoning": {"effort": "medium"},
    }
    result = convert_request(req)
    assert result["thinking"] == {"type": "enabled"}


def test_reasoning_effort_high_to_enabled():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "reasoning": {"effort": "high"},
    }
    result = convert_request(req)
    assert result["thinking"] == {"type": "enabled"}


def test_reasoning_effort_low_to_enabled():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "reasoning": {"effort": "low"},
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


def test_text_format_openai_style_json_schema_rewritten_for_kimi():
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "greeting",
                "schema": {"type": "object"},
                "strict": True,
            },
        },
    }
    result = convert_request(req)
    assert result["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "greeting",
            "schema": {"type": "object"},
            "strict": True,
        },
    }


def test_tools_adaptation():
    """Responses API format tools (flat) are converted to Chat Completions format (nested)."""
    req = {
        "model": "kimi-k2.6",
        "input": "Hello!",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object"},
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


def test_empty_assistant_message_filtered():
    req = {
        "model": "kimi-k2.6",
        "input": [
            {"role": "assistant", "content": "", "reasoning_content": "hidden chain of thought"},
            {"role": "user", "content": "Hello!"},
        ],
    }
    result = convert_request(req)
    assert result["messages"] == [{"role": "user", "content": "Hello!"}]


def test_role_tool_with_call_id_mapped_to_tool_call_id():
    req = {
        "model": "kimi-k2.6",
        "input": [
            {"role": "tool", "call_id": "shell_command:145", "content": "done"},
        ],
    }
    result = convert_request(req)
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "shell_command:145"
    assert result["messages"][0]["content"] == "done"


def test_function_call_output_content_parts_coerced_to_text():
    req = {
        "model": "kimi-k2.6",
        "input": [
            {
                "type": "function_call_output",
                "call_id": "shell_command:145",
                "output": [{"type": "output_text", "text": "file.txt"}],
            },
        ],
    }
    result = convert_request(req)
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "shell_command:145"
    assert result["messages"][0]["content"] == "file.txt"
