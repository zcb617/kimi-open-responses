"""Convert responses API requests to chat.completions format."""


def convert_request(responses_req: dict) -> dict:
    """Convert a responses API request dict to chat.completions format."""
    chat_req: dict = {"model": responses_req["model"]}

    # Handle input -> messages
    input_data = responses_req.get("input", "")
    if isinstance(input_data, str):
        chat_req["messages"] = [{"role": "user", "content": input_data}]
    elif isinstance(input_data, list):
        chat_req["messages"] = list(input_data)

    # Handle instructions -> system message
    instructions = responses_req.get("instructions")
    if instructions:
        chat_req["messages"].insert(0, {"role": "system", "content": instructions})

    # Parameter mapping
    if "temperature" in responses_req:
        chat_req["temperature"] = responses_req["temperature"]
    if "max_output_tokens" in responses_req:
        chat_req["max_tokens"] = responses_req["max_output_tokens"]
    if "top_p" in responses_req:
        chat_req["top_p"] = responses_req["top_p"]
    if "presence_penalty" in responses_req:
        chat_req["presence_penalty"] = responses_req["presence_penalty"]
    if "frequency_penalty" in responses_req:
        chat_req["frequency_penalty"] = responses_req["frequency_penalty"]
    if "tool_choice" in responses_req:
        chat_req["tool_choice"] = responses_req["tool_choice"]
    if "stream" in responses_req:
        chat_req["stream"] = responses_req["stream"]

    # Reasoning effort -> thinking.type
    reasoning = responses_req.get("reasoning")
    if reasoning:
        effort = reasoning.get("effort", "medium")
        if effort == "none":
            chat_req["thinking"] = {"type": "disabled"}
        else:
            chat_req["thinking"] = {"type": "enabled"}

    # text.format -> response_format
    text_config = responses_req.get("text")
    if text_config and "format" in text_config:
        fmt = text_config["format"]
        chat_req["response_format"] = dict(fmt)

    # Tools — pass through (already compatible)
    if "tools" in responses_req:
        chat_req["tools"] = responses_req["tools"]

    return chat_req
