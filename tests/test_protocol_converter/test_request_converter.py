from src.openai_protocol_converter.request_converter import (
    _CUSTOM_TOOL_PROXY_PREFIX,
    _NAMESPACE_TOOL_PROXY_PREFIX,
    convert_request,
)


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
    """Current Codex additional_tools function declarations reach Chat format."""
    req = {
        "model": "kimi-k3",
        "input": [{
            "type": "additional_tools",
            "tools": [{"type": "namespace", "name": "weather", "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object"},
            },
            ]}],
        }],
    }
    result = convert_request(req)
    assert len(result["tools"]) == 1
    assert result["tools"][0]["type"] == "function"
    assert result["tools"][0]["function"]["name"].startswith(_NAMESPACE_TOOL_PROXY_PREFIX)


def test_additional_tools_namespaces_are_flattened_for_chat():
    """Codex's current nested namespace tool shape must reach Kimi."""
    req = {
        "model": "kimi-k3",
        "input": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "run it"}]},
            {
                "type": "additional_tools",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "shell",
                        "tools": [
                            {
                                "type": "function",
                                "name": "run",
                                "description": "Run a command",
                                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
                            },
                            {
                                "type": "custom",
                                "name": "exec",
                                "description": "Execute free-form input",
                                "format": {"type": "text"},
                            },
                        ],
                    }
                ],
            },
        ],
    }

    result = convert_request(req)

    assert len(result["tools"]) == 2
    names = {tool["function"]["name"] for tool in result["tools"]}
    assert any(name.startswith(_NAMESPACE_TOOL_PROXY_PREFIX) for name in names)
    custom_name = next(name for name in names if name.startswith(_CUSTOM_TOOL_PROXY_PREFIX))
    custom = next(tool for tool in result["tools"] if tool["function"]["name"] == custom_name)
    assert custom["function"]["description"] == "Execute free-form input"
    assert custom["function"]["parameters"]["required"] == ["input"]


def test_custom_tool_history_is_converted_to_chat_tool_messages():
    req = {
        "model": "kimi-k3",
        "input": [
            {
                "type": "additional_tools",
                "tools": [{"type": "namespace", "name": "shell", "tools": [
                    {"type": "custom", "name": "exec", "format": {"type": "text"}},
                ]}],
            },
            {"type": "custom_tool_call", "call_id": "call_exec", "name": "exec", "input": "ls -la"},
            {"type": "custom_tool_call_output", "call_id": "call_exec", "output": "done"},
        ],
    }

    result = convert_request(req)

    assert result["messages"][0]["role"] == "assistant"
    tool_call = result["messages"][0]["tool_calls"][0]
    assert tool_call["id"] == "call_exec"
    assert tool_call["function"]["name"].startswith(_CUSTOM_TOOL_PROXY_PREFIX)
    assert tool_call["function"]["arguments"] == '{"input": "ls -la"}'
    assert result["messages"][1] == {
        "role": "tool",
        "tool_call_id": "call_exec",
        "content": "done",
    }


def test_namespace_proxy_names_are_stable_for_same_tool_names():
    req = {
        "model": "kimi-k3",
        "input": [{
            "type": "additional_tools",
            "tools": [
                {"type": "namespace", "name": "one", "tools": [{"type": "function", "name": "run"}]},
                {"type": "namespace", "name": "two", "tools": [{"type": "function", "name": "run"}]},
            ],
        }],
    }

    first = convert_request(req)
    second = convert_request(req)
    first_names = [tool["function"]["name"] for tool in first["tools"]]
    second_names = [tool["function"]["name"] for tool in second["tools"]]
    assert first_names == second_names
    assert len(set(first_names)) == 2


def test_namespaced_function_call_history_uses_namespace_proxy():
    req = {
        "model": "kimi-k3",
        "input": [
            {
                "type": "additional_tools",
                "tools": [{"type": "namespace", "name": "collaboration", "tools": [
                    {"type": "function", "name": "interrupt_agent"},
                ]}],
            },
            {
                "type": "function_call",
                "call_id": "call_interrupt",
                "name": "interrupt_agent",
                "namespace": "collaboration",
                "arguments": "{}",
            },
        ],
    }

    result = convert_request(req)

    function_name = result["messages"][0]["tool_calls"][0]["function"]["name"]
    assert function_name.startswith(_NAMESPACE_TOOL_PROXY_PREFIX)
    assert len(function_name) <= 64


def test_real_mcp_namespace_proxy_names_fit_chat_function_limit():
    req = {
        "model": "kimi-k3",
        "input": [{
            "type": "additional_tools",
            "tools": [{"type": "namespace", "name": "mcp__fastctx", "tools": [
                {"type": "function", "name": "inspect_local_file"},
                {"type": "custom", "name": "exec", "format": {"type": "text"}},
            ]}],
        }],
    }

    result = convert_request(req)

    names = [tool["function"]["name"] for tool in result["tools"]]
    assert names
    assert all(len(name) <= 64 for name in names)


def test_current_two_namespace_fixture_expands_all_seven_tools():
    namespaces = [
        {"type": "namespace", "name": "collaboration", "tools": [
            {"type": "function", "name": name} for name in (
                "spawn_agent", "send_message", "wait_agent", "list_agents", "read_agent"
            )
        ]},
        {"type": "namespace", "name": "mcp__fastctx", "tools": [
            {"type": "function", "name": "inspect_local_file"},
            {"type": "custom", "name": "exec", "format": {"type": "text"}},
        ]},
    ]
    result = convert_request({
        "model": "kimi-k3",
        "input": [{"type": "additional_tools", "tools": namespaces}],
    })
    assert len(result["tools"]) == 7


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


def test_function_call_output_accepts_tool_call_id_field():
    req = {
        "model": "kimi-k2.6",
        "input": [
            {
                "type": "function_call_output",
                "tool_call_id": "shell_command:73",
                "output": "ok",
            },
        ],
    }
    result = convert_request(req)
    assert result["messages"][0]["role"] == "tool"
    assert result["messages"][0]["tool_call_id"] == "shell_command:73"
    assert result["messages"][0]["content"] == "ok"


def test_function_call_accepts_tool_call_id_field():
    req = {
        "model": "kimi-k2.6",
        "input": [
            {
                "type": "function_call",
                "tool_call_id": "shell_command:73",
                "name": "run_shell",
                "arguments": "{\"cmd\":\"pwd\"}",
            },
        ],
    }
    result = convert_request(req)
    assert result["messages"][0]["role"] == "assistant"
    assert result["messages"][0]["tool_calls"][0]["id"] == "shell_command:73"


def test_assistant_tool_calls_without_id_backfilled_from_tool_messages():
    req = {
        "model": "kimi-k2.6",
        "input": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "shell_command", "arguments": "{\"cmd\":\"pwd\"}"},
                    },
                    {
                        "type": "function",
                        "function": {"name": "shell_command", "arguments": "{\"cmd\":\"ls\"}"},
                    },
                ],
            },
            {"role": "developer", "content": "keep this as system message"},
            {"role": "tool", "tool_call_id": "tool_A", "content": "pwd output"},
            {"role": "tool", "call_id": "tool_B", "content": "ls output"},
        ],
    }
    result = convert_request(req)
    messages = result["messages"]
    assert messages[0]["role"] == "assistant"
    assert [tc["id"] for tc in messages[0]["tool_calls"]] == ["tool_A", "tool_B"]
    assert messages[1] == {"role": "tool", "tool_call_id": "tool_A", "content": "pwd output"}
    assert messages[2] == {"role": "tool", "tool_call_id": "tool_B", "content": "ls output"}
    assert messages[3] == {"role": "system", "content": "keep this as system message"}


def test_assistant_tool_calls_mixed_ids_keep_sequence():
    req = {
        "model": "kimi-k2.6",
        "input": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "tool_X",
                        "type": "function",
                        "function": {"name": "shell_command", "arguments": "{\"cmd\":\"echo x\"}"},
                    },
                    {
                        "type": "function",
                        "function": {"name": "shell_command", "arguments": "{\"cmd\":\"echo y\"}"},
                    },
                ],
            },
            {"role": "tool", "tool_call_id": "tool_X", "content": "x"},
            {"role": "tool", "tool_call_id": "tool_Y", "content": "y"},
        ],
    }
    result = convert_request(req)
    messages = result["messages"]
    assert messages[0]["role"] == "assistant"
    assert [tc["id"] for tc in messages[0]["tool_calls"]] == ["tool_X", "tool_Y"]
    assert messages[1] == {"role": "tool", "tool_call_id": "tool_X", "content": "x"}
    assert messages[2] == {"role": "tool", "tool_call_id": "tool_Y", "content": "y"}
