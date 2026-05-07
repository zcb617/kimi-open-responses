"""Type definitions for OpenAI protocol converter."""
from typing import TypedDict, NotRequired


# --- chat.completions types ---

class ChatMessage(TypedDict):
    role: str
    content: str


class ChatToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: dict


class ChatTool(TypedDict):
    type: str
    function: ChatToolFunction


class ChatCompletionRequest(TypedDict):
    model: str
    messages: list[ChatMessage]
    temperature: NotRequired[float]
    max_tokens: NotRequired[int]
    top_p: NotRequired[float]
    presence_penalty: NotRequired[float]
    frequency_penalty: NotRequired[float]
    tools: NotRequired[list[ChatTool]]
    tool_choice: NotRequired[str | dict]
    stream: NotRequired[bool]
    response_format: NotRequired[dict]
    thinking: NotRequired[dict]


class ChatCompletionChoice(TypedDict):
    index: int
    message: dict
    finish_reason: str | None


class ChatCompletionResponse(TypedDict):
    id: str
    object: str
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict


# --- responses API types ---

class ResponsesInputText(TypedDict):
    type: str
    text: str


class ResponsesInputMessage(TypedDict):
    role: str
    content: str | list[ResponsesInputText]


class ResponsesTextFormat(TypedDict):
    type: str
    json_schema: NotRequired[dict]


class ResponsesTextConfig(TypedDict):
    format: NotRequired[ResponsesTextFormat]


class ResponsesToolFunction(TypedDict):
    name: str
    description: NotRequired[str]
    parameters: NotRequired[dict]
    strict: NotRequired[bool]


class ResponsesTool(TypedDict):
    type: str
    function: ResponsesToolFunction


class ResponsesReasoning(TypedDict):
    effort: NotRequired[str]


class ResponsesRequest(TypedDict):
    model: str
    input: str | list[ResponsesInputMessage]
    instructions: NotRequired[str]
    temperature: NotRequired[float]
    max_output_tokens: NotRequired[int]
    top_p: NotRequired[float]
    presence_penalty: NotRequired[float]
    frequency_penalty: NotRequired[float]
    tools: NotRequired[list[ResponsesTool]]
    tool_choice: NotRequired[str | dict]
    stream: NotRequired[bool]
    text: NotRequired[ResponsesTextConfig]
    reasoning: NotRequired[ResponsesReasoning]
    previous_response_id: NotRequired[str]


class ResponsesOutputText(TypedDict):
    type: str
    text: str


class ResponsesOutputFunctionCall(TypedDict):
    type: str
    call_id: str
    name: str
    arguments: str


class ResponsesOutputItem(TypedDict):
    type: str
    role: NotRequired[str]
    content: NotRequired[list[ResponsesOutputText | ResponsesOutputFunctionCall]]


class ResponsesResponse(TypedDict):
    id: str
    object: str
    created_at: int
    model: str
    output: list[ResponsesOutputItem]
    usage: dict
    status: NotRequired[str]
