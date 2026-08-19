"""Convert responses API requests to chat.completions format."""

import base64
import json


# Prefix is part of the wire-level compatibility contract with stream_converter.
# It keeps namespace collisions reversible without changing src/proxy.py.
_CUSTOM_TOOL_PROXY_PREFIX = "__cf1_"
_NAMESPACE_TOOL_PROXY_PREFIX = "__nf1_"


_INPUT_TEXT = "input_text"
_OUTPUT_TEXT = "output_text"
_INPUT_IMAGE = "input_image"
_REFUSAL = "refusal"


def _encode_tool_proxy(prefix: str, namespace: str, name: str) -> str:
    """Encode namespace and tool name into a deterministic Chat tool name."""
    payload = json.dumps([namespace, name], ensure_ascii=False, separators=(",", ":"))
    token = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{prefix}{token}"


def _custom_tool_parameters(tool: dict) -> dict:
    """Expose a custom tool as one JSON string argument to Chat Completions."""
    # The custom Responses input is free-form text; wrapping it in one required
    # property gives Kimi a valid JSON-schema function declaration.
    input_description = "Raw custom-tool input. Put the complete free-form payload in this string."
    tool_format = tool.get("format")
    if isinstance(tool_format, dict) and tool_format.get("type") == "grammar":
        grammar = tool_format.get("definition")
        if isinstance(grammar, str) and grammar:
            syntax = tool_format.get("syntax", "grammar")
            input_description += f" The string must match this {syntax} grammar:\n{grammar}"
    if tool.get("name") == "apply_patch":
        input_description += (
            " The input must start with '*** Begin Patch' and end with '*** End Patch'."
            " In an added line, '+' is a syntax marker and the following characters become"
            " file content; do not add a separator space after '+' unless that space belongs"
            " in the file."
        )

    return {
        "type": "object",
        "properties": {"input": {"type": "string", "description": input_description}},
        "required": ["input"],
        "additionalProperties": False,
    }


def _iter_additional_tools(input_data) -> list[dict]:
    """Yield namespace tool definitions embedded in input additional_tools."""
    if not isinstance(input_data, list):
        return []
    result: list[dict] = []
    for item in input_data:
        if not isinstance(item, dict) or item.get("type") != "additional_tools":
            continue
        namespaces = item.get("tools", [])
        if not isinstance(namespaces, list):
            continue
        for namespace_item in namespaces:
            if not isinstance(namespace_item, dict):
                continue
            namespace = namespace_item.get("namespace") or namespace_item.get("name") or ""
            nested = namespace_item.get("tools", [])
            if not isinstance(nested, list):
                continue
            for tool in nested:
                if isinstance(tool, dict):
                    result.append({"namespace": str(namespace), "tool": tool})
    return result


def _convert_additional_tool(tool_entry: dict) -> tuple[dict | None, str | None]:
    """Convert one namespaced Responses tool and return (Chat tool, kind)."""
    namespace = tool_entry.get("namespace", "")
    tool = tool_entry.get("tool", {})
    tool_type = tool.get("type")
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        return None, None
    if tool_type == "function":
        # Namespace function names use the same reversible encoding as custom
        # tools; punctuation in a namespace must not violate Chat name rules.
        proxy_name = _encode_tool_proxy(_NAMESPACE_TOOL_PROXY_PREFIX, namespace, name)
        function_def = {
            key: tool[key]
            for key in ("description", "parameters", "strict")
            if key in tool
        }
        function_def["name"] = proxy_name
        return {"type": "function", "function": function_def}, "function"
    if tool_type == "custom":
        proxy_name = _encode_tool_proxy(_CUSTOM_TOOL_PROXY_PREFIX, namespace, name)
        function_def = {
            "name": proxy_name,
            "description": tool.get("description", ""),
            "parameters": _custom_tool_parameters(tool),
        }
        return {"type": "function", "function": function_def}, "custom"
    return None, None


def _convert_top_level_tool(tool: dict) -> dict | None:
    """Convert current Codex function/custom tools to Chat Completions format."""
    if not isinstance(tool, dict):
        return None

    name = tool.get("name")
    if not isinstance(name, str) or not name:
        return None

    if tool.get("type") == "custom":
        return {
            "type": "function",
            "function": {
                "name": _encode_tool_proxy(_CUSTOM_TOOL_PROXY_PREFIX, "", name),
                "description": tool.get("description", ""),
                "parameters": _custom_tool_parameters(tool),
            },
        }

    if tool.get("type") != "function":
        return None

    function_def = {
        "name": name,
        "parameters": tool.get("parameters") or {"type": "object", "properties": {}},
    }
    description = tool.get("description")
    if isinstance(description, str):
        function_def["description"] = description
    if isinstance(tool.get("strict"), bool):
        function_def["strict"] = tool["strict"]

    return {"type": "function", "function": function_def}


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


def _convert_message(
    msg: dict,
    custom_name_map: dict[str, str] | None = None,
    namespace_name_map: dict[tuple[str, str], str] | None = None,
) -> dict | None:
    """Convert a single Responses API message to Chat Completions format."""
    msg_type = msg.get("type", "")

    custom_name_map = custom_name_map or {}
    namespace_name_map = namespace_name_map or {}
    if msg_type in ("custom_tool_call_output", "function_call_output"):
        output = _convert_content(msg.get("output", ""))
        tool_call_id = msg.get("tool_call_id") or msg.get("call_id") or msg.get("id", "")
        return {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": _coerce_text_content(output),
        }

    if msg_type in ("custom_tool_call", "function_call"):
        reasoning_content = msg.get("reasoning_content", "")
        if not isinstance(reasoning_content, str):
            reasoning_content = ""
        call_id = msg.get("call_id") or msg.get("tool_call_id") or msg.get("id", "")
        name = msg.get("name", "")
        namespace = msg.get("namespace")
        if msg_type == "function_call" and isinstance(namespace, str) and isinstance(name, str):
            name = namespace_name_map.get(
                (namespace, name), _encode_tool_proxy(_NAMESPACE_TOOL_PROXY_PREFIX, namespace, name)
            )
        # Custom tool input is free-form, while Chat requires a function
        # arguments string. Preserve raw input and use the current tool map
        # when this history item has a matching custom declaration.
        if msg_type == "custom_tool_call":
            name = custom_name_map.get(name, name)
            if "input" in msg:
                raw_input = msg.get("input", "")
                if not isinstance(raw_input, str):
                    raw_input = _coerce_text_content(raw_input)
                arguments = json.dumps({"input": raw_input}, ensure_ascii=False)
            else:
                arguments = msg.get("arguments", "")
        else:
            arguments = msg.get("arguments", "")
        return {
            "role": "assistant",
            "content": None,
            "reasoning_content": reasoning_content,
            "tool_calls": [{
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": arguments,
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
    if role == "assistant" and isinstance(result.get("tool_calls"), list):
        for tool_call in result["tool_calls"]:
            if not isinstance(tool_call, dict):
                continue
            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue
            namespace = tool_call.get("namespace") or function.get("namespace")
            name = function.get("name")
            if isinstance(namespace, str) and isinstance(name, str):
                function["name"] = namespace_name_map.get(
                    (namespace, name), _encode_tool_proxy(_NAMESPACE_TOOL_PROXY_PREFIX, namespace, name)
                )
    return result


def convert_request(responses_req: dict) -> dict:
    """Convert a responses API request dict to chat.completions format."""
    chat_req: dict = {"model": responses_req["model"]}

    input_data = responses_req.get("input", "")
    additional_tool_entries = _iter_additional_tools(input_data)
    custom_name_map = {}
    namespace_name_map = {}
    for entry in additional_tool_entries:
        tool = entry["tool"]
        if tool.get("type") == "function" and isinstance(tool.get("name"), str):
            namespace_name_map[(entry["namespace"], tool["name"])] = _encode_tool_proxy(
                _NAMESPACE_TOOL_PROXY_PREFIX, entry["namespace"], tool["name"]
            )
        if tool.get("type") == "custom" and isinstance(tool.get("name"), str):
            # A name is mapped only when unambiguous; duplicate names remain
            # caller-provided names and cannot be inferred from history alone.
            custom_name_map.setdefault(tool["name"], []).append(entry)
    custom_name_map = {
        name: _encode_tool_proxy(_CUSTOM_TOOL_PROXY_PREFIX, entries[0]["namespace"], name)
        for name, entries in custom_name_map.items()
        if len(entries) == 1
    }
    if isinstance(input_data, str):
        chat_req["messages"] = [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        converted_msgs = []
        for m in input_data:
            cm = _convert_message(m, custom_name_map, namespace_name_map)
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

    if "stream" in responses_req:
        chat_req["stream"] = responses_req["stream"]
    if "max_output_tokens" in responses_req:
        # Kimi Chat API deprecates max_tokens in favor of max_completion_tokens.
        chat_req["max_completion_tokens"] = responses_req["max_output_tokens"]

    response_format = _convert_response_format(responses_req.get("text"))
    if response_format is not None:
        chat_req["response_format"] = response_format

    for tool in responses_req.get("tools", []):
        converted_tool = _convert_top_level_tool(tool)
        if converted_tool is not None:
            chat_req.setdefault("tools", []).append(converted_tool)

    # Codex Desktop namespace tools are nested in input.additional_tools.
    for entry in additional_tool_entries:
        converted_tool, _kind = _convert_additional_tool(entry)
        if converted_tool is not None:
            chat_req.setdefault("tools", []).append(converted_tool)

    if "tool_choice" in responses_req:
        tc = responses_req["tool_choice"]
        chat_req["tool_choice"] = "none" if tc == "none" else "auto"

    # K2.7 Code requires thinking mode and rejects disabled thinking.
    chat_req["thinking"] = {"type": "enabled"}
    return chat_req
