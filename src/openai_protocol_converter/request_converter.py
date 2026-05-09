"""Convert responses API requests to chat.completions format."""


_INPUT_TEXT = "input_text"
_OUTPUT_TEXT = "output_text"
_INPUT_IMAGE = "input_image"
_REFUSAL = "refusal"


def _convert_content_part(part: dict) -> dict | None:
    """Convert a single Responses API content part to Chat Completions format."""
    part_type = part.get("type", "")
    if part_type == _INPUT_TEXT or part_type == _OUTPUT_TEXT:
        return {"type": "text", "text": part.get("text", "")}
    if part_type == _INPUT_IMAGE:
        image_url = part.get("image_url", "")
        if isinstance(image_url, str):
            return {"type": "image_url", "image_url": {"url": image_url}}
        if isinstance(image_url, dict):
            return {"type": "image_url", "image_url": image_url}
    if part_type == _REFUSAL:
        return None
    if part_type in ("text", "image_url", "video_url"):
        return part
    return None


def _convert_content(content):
    """Convert Responses API content (string or part list) to Chat Completions format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        converted = []
        all_text = True
        for part in content:
            if not isinstance(part, dict):
                continue
            cp = _convert_content_part(part)
            if cp:
                converted.append(cp)
                if cp.get("type") != "text":
                    all_text = False
        if converted and all_text:
            return "".join(p.get("text", "") for p in converted)
        return converted
    return content


def _is_empty_content(content) -> bool:
    """Whether message content is effectively empty."""
    if content is None:
        return True
    if isinstance(content, str):
        return content.strip() == ""
    if isinstance(content, list):
        return len(content) == 0
    return False


def _convert_response_format(text_config: dict) -> dict | None:
    """Convert Responses text.format to Kimi Chat response_format."""
    if not isinstance(text_config, dict):
        return None
    fmt = text_config.get("format")
    if not isinstance(fmt, dict):
        return None

    fmt_type = fmt.get("type")
    if fmt_type != "json_schema":
        return dict(fmt)

    # Kimi expects: {"type": "json_schema", "json_schema": {...}}
    nested_schema = fmt.get("json_schema")
    if isinstance(nested_schema, dict):
        return {
            "type": "json_schema",
            "json_schema": dict(nested_schema),
        }

    json_schema = {}
    for key in ("name", "schema", "description", "strict"):
        if key in fmt:
            json_schema[key] = fmt[key]
    if not json_schema:
        # Keep original payload if we cannot safely reconstruct.
        return dict(fmt)
    return {
        "type": "json_schema",
        "json_schema": json_schema,
    }


def _convert_message(msg: dict) -> dict | None:
    """Convert a single Responses API message to Chat Completions format."""
    msg_type = msg.get("type", "")

    if msg_type == "function_call_output":
        return {
            "role": "tool",
            "tool_call_id": msg.get("call_id") or msg.get("id", ""),
            "content": msg.get("output", ""),
        }

    if msg_type == "function_call":
        reasoning_content = msg.get("reasoning_content", "")
        if not isinstance(reasoning_content, str):
            reasoning_content = ""
        return {
            "role": "assistant",
            "content": None,
            "reasoning_content": reasoning_content,
            "tool_calls": [{
                "id": msg.get("call_id") or msg.get("id", ""),
                "type": "function",
                "function": {
                    "name": msg.get("name", ""),
                    "arguments": msg.get("arguments", ""),
                },
            }],
        }

    result = {}
    role = msg.get("role", "user")
    if role == "developer":
        role = "system"
    result["role"] = role
    result["content"] = _convert_content(msg.get("content"))
    for key in ("name", "tool_calls", "tool_call_id", "reasoning_content"):
        if key in msg:
            result[key] = msg[key]
    return result


def convert_request(responses_req: dict) -> dict:
    """Convert a responses API request dict to chat.completions format."""
    chat_req: dict = {"model": responses_req["model"]}

    input_data = responses_req.get("input", "")
    if isinstance(input_data, str):
        chat_req["messages"] = [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        converted_msgs = []
        for m in input_data:
            cm = _convert_message(m)
            if cm is None:
                continue
            role = cm.get("role", "")
            content = cm.get("content")
            if role in ("user", "system") and _is_empty_content(content):
                continue
            # Kimi rejects empty assistant text messages; keep only assistant
            # messages with content or tool_calls.
            if role == "assistant" and not cm.get("tool_calls") and _is_empty_content(content):
                continue
            # Merge consecutive assistant tool_calls into a single message
            if (role == "assistant"
                    and cm.get("tool_calls")
                    and converted_msgs
                    and converted_msgs[-1].get("role") == "assistant"
                    and converted_msgs[-1].get("tool_calls")):
                converted_msgs[-1]["tool_calls"].extend(cm["tool_calls"])
                continue
            converted_msgs.append(cm)
        chat_req["messages"] = converted_msgs

    instructions = responses_req.get("instructions")
    if instructions:
        chat_req["messages"].insert(0, {"role": "system", "content": instructions})

    for key in ("temperature", "top_p", "presence_penalty", "frequency_penalty", "stream"):
        if key in responses_req:
            chat_req[key] = responses_req[key]
    if "max_output_tokens" in responses_req:
        # Kimi Chat API deprecates max_tokens in favor of max_completion_tokens.
        chat_req["max_completion_tokens"] = responses_req["max_output_tokens"]

    response_format = _convert_response_format(responses_req.get("text"))
    if response_format is not None:
        chat_req["response_format"] = response_format

    # Tools — Responses API format differs from Chat Completions
    if "tools" in responses_req:
        tools = responses_req["tools"]
        if isinstance(tools, list):
            chat_req["tools"] = []
            for tool in tools:
                if tool.get("type") == "function":
                    function_def = {}
                    for key in ("name", "description", "parameters", "strict"):
                        if key in tool:
                            function_def[key] = tool[key]
                    chat_req["tools"].append({
                        "type": "function",
                        "function": function_def,
                    })
            if not chat_req["tools"]:
                del chat_req["tools"]

    if "tool_choice" in responses_req:
        tc = responses_req["tool_choice"]
        if tc == "required":
            # Kimi currently does not support tool_choice=required.
            chat_req["tool_choice"] = "auto"
            tc = None
        if isinstance(tc, dict) and tc.get("type") == "function" and "name" in tc:
            chat_req["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }
        elif tc is not None:
            chat_req["tool_choice"] = tc

    # Kimi-specific: thinking parameter
    reasoning = responses_req.get("reasoning")
    if reasoning:
        effort = reasoning.get("effort", "medium")
        if effort == "none":
            chat_req["thinking"] = {"type": "disabled"}
        else:
            chat_req["thinking"] = {"type": "enabled"}
    # If history already carries reasoning_content, enable preserved thinking for K2.6.
    has_reasoning_content = any(
        isinstance(msg, dict) and msg.get("reasoning_content")
        for msg in chat_req.get("messages", [])
    )
    if has_reasoning_content and str(chat_req.get("model", "")).startswith("kimi-k2.6"):
        thinking = chat_req.get("thinking")
        if not isinstance(thinking, dict):
            chat_req["thinking"] = {"type": "enabled", "keep": "all"}
        elif thinking.get("type") == "enabled":
            chat_req["thinking"].setdefault("keep", "all")

    return chat_req
