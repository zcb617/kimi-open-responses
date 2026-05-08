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
    if part_type in ("text", "image_url"):
        return part
    return None


def _convert_content(content):
    """Convert Responses API content (string or part list) to Chat Completions format."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        converted = []
        for part in content:
            if not isinstance(part, dict):
                continue
            cp = _convert_content_part(part)
            if cp:
                converted.append(cp)
        if converted and all(p.get("type") == "text" for p in converted):
            return "".join(p.get("text", "") for p in converted)
        if len(converted) == 1 and converted[0].get("type") == "text":
            return converted[0]["text"]
        return converted
    return content


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
        return {
            "role": "assistant",
            "content": None,
            "reasoning_content": "",
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
    for key in ("name", "tool_calls", "tool_call_id"):
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
            role = cm.get("role", "")
            content = cm.get("content")
            if role in ("user", "system") and (content is None or content == "" or content == []):
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

    for key in ("temperature", "max_output_tokens", "top_p",
                "presence_penalty", "frequency_penalty", "tool_choice", "stream"):
        if key in responses_req:
            chat_req[key if key != "max_output_tokens" else "max_tokens"] = responses_req[key]

    text_config = responses_req.get("text")
    if text_config and "format" in text_config:
        chat_req["response_format"] = dict(text_config["format"])

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
                elif tool.get("type") == "plugin":
                    chat_req["tools"].append(tool)
            if not chat_req["tools"]:
                del chat_req["tools"]

    if "tool_choice" in responses_req:
        tc = responses_req["tool_choice"]
        if isinstance(tc, dict) and tc.get("type") == "function" and "name" in tc:
            chat_req["tool_choice"] = {
                "type": "function",
                "function": {"name": tc["name"]},
            }
        else:
            chat_req["tool_choice"] = tc

    # Kimi-specific: thinking parameter
    reasoning = responses_req.get("reasoning")
    if reasoning:
        effort = reasoning.get("effort", "medium")
        if effort == "none":
            chat_req["thinking"] = {"type": "disabled"}
        else:
            chat_req["thinking"] = {"type": "enabled"}

    return chat_req
