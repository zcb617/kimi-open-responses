"""Convert responses API requests to chat.completions format."""


_INPUT_TEXT = "input_text"
_OUTPUT_TEXT = "output_text"
_INPUT_IMAGE = "input_image"
_REFUSAL = "refusal"


def _coerce_text_content(content) -> str:
    """Coerce mixed content shapes to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if not isinstance(part, dict):
                continue
            part_type = part.get("type", "")
            if part_type in (_INPUT_TEXT, _OUTPUT_TEXT, "text"):
                text = part.get("text", "")
                if isinstance(text, str):
                    chunks.append(text)
            elif part_type == _REFUSAL:
                refusal = part.get("refusal", "")
                if isinstance(refusal, str):
                    chunks.append(refusal)
        return "".join(chunks)
    return str(content)


def _normalize_tool_call_sequences(messages: list[dict]) -> list[dict]:
    """Ensure each assistant tool_calls block is immediately followed by matching tool messages."""
    non_tool_messages: list[dict] = []
    tool_messages_by_id: dict[str, list[dict]] = {}
    tool_messages_in_order: list[dict] = []

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        if msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id") or msg.get("call_id") or msg.get("id", "")
            if not tool_call_id:
                continue
            normalized_tool = dict(msg)
            normalized_tool["tool_call_id"] = tool_call_id
            normalized_tool["content"] = _coerce_text_content(normalized_tool.get("content", ""))
            tool_messages_by_id.setdefault(tool_call_id, []).append(normalized_tool)
            tool_messages_in_order.append(normalized_tool)
            continue
        non_tool_messages.append(msg)

    consumed_tool_ids: set[int] = set()
    ordered_tool_cursor = 0

    def _mark_consumed(tool_msg: dict):
        consumed_tool_ids.add(id(tool_msg))

    def _pop_tool_by_id(call_id: str) -> dict | None:
        queue = tool_messages_by_id.get(call_id)
        if not queue:
            return None
        while queue:
            candidate = queue.pop(0)
            if id(candidate) in consumed_tool_ids:
                continue
            _mark_consumed(candidate)
            if not queue:
                tool_messages_by_id.pop(call_id, None)
            return candidate
        tool_messages_by_id.pop(call_id, None)
        return None

    def _pop_next_tool_in_order() -> dict | None:
        nonlocal ordered_tool_cursor
        while ordered_tool_cursor < len(tool_messages_in_order):
            candidate = tool_messages_in_order[ordered_tool_cursor]
            ordered_tool_cursor += 1
            if id(candidate) in consumed_tool_ids:
                continue
            _mark_consumed(candidate)
            return candidate
        return None

    normalized: list[dict] = []
    assistant_tool_blocks = 0
    for msg in non_tool_messages:
        msg_out = dict(msg)
        normalized.append(msg_out)

        if msg_out.get("role") != "assistant":
            continue
        tool_calls = msg_out.get("tool_calls")
        if not isinstance(tool_calls, list) or not tool_calls:
            continue
        assistant_tool_blocks += 1

        updated_tool_calls = []
        for idx, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            tc = dict(tool_call)
            call_id = tc.get("id") or tc.get("call_id") or tc.get("tool_call_id") or ""
            matched_tool: dict | None = None
            if call_id:
                call_id = str(call_id)
                matched_tool = _pop_tool_by_id(call_id)
            if not call_id:
                matched_tool = _pop_next_tool_in_order()
                if matched_tool is not None:
                    call_id = matched_tool.get("tool_call_id", "")
            if not call_id:
                fn_name = ""
                function = tc.get("function")
                if isinstance(function, dict):
                    fn_name = function.get("name", "")
                call_id = f"{fn_name or 'tool'}:{idx}"
            if not matched_tool:
                matched_tool = _pop_tool_by_id(call_id)
            tc["id"] = call_id
            updated_tool_calls.append(tc)

            if matched_tool:
                normalized.append(matched_tool)
            else:
                normalized.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": "",
                })

        msg_out["tool_calls"] = updated_tool_calls

    if assistant_tool_blocks == 0:
        return non_tool_messages + tool_messages_in_order

    return normalized


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
        output = _convert_content(msg.get("output", ""))
        tool_call_id = msg.get("tool_call_id") or msg.get("call_id") or msg.get("id", "")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _coerce_text_content(output),
        }

    if msg_type == "function_call":
        reasoning_content = msg.get("reasoning_content", "")
        if not isinstance(reasoning_content, str):
            reasoning_content = ""
        call_id = msg.get("call_id") or msg.get("tool_call_id") or msg.get("id", "")
        return {
            "role": "assistant",
            "content": None,
            "reasoning_content": reasoning_content,
            "tool_calls": [{
                "id": call_id,
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
    if role == "tool":
        # Compatibility: some callers use call_id/id on tool messages.
        if "tool_call_id" not in msg:
            tool_call_id = msg.get("call_id") or msg.get("id", "")
            if tool_call_id:
                result["tool_call_id"] = tool_call_id
        result["content"] = _coerce_text_content(result.get("content"))
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
        chat_req["messages"] = _normalize_tool_call_sequences(converted_msgs)

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
