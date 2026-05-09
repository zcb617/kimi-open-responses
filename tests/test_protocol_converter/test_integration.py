"""End-to-end integration tests for the protocol converter."""
import json

from src.openai_protocol_converter import convert_request, convert_response, StreamConverter


def test_full_text_request_response_cycle():
    responses_req = {
        "model": "kimi-k2.6",
        "input": "What is the weather?",
        "temperature": 0.5,
        "max_output_tokens": 50,
    }

    chat_req = convert_request(responses_req)
    assert chat_req["model"] == "kimi-k2.6"
    assert chat_req["messages"] == [{"role": "user", "content": "What is the weather?"}]
    assert chat_req["temperature"] == 0.5
    assert chat_req["max_completion_tokens"] == 50

    chat_resp = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1700000100,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "I don't have real-time weather data.",
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }

    responses_resp = convert_response(chat_resp)
    assert responses_resp["id"] == "chatcmpl-test"
    assert responses_resp["object"] == "response"
    assert responses_resp["created_at"] == 1700000100
    assert responses_resp["output"][0]["content"][0]["text"] == "I don't have real-time weather data."
    assert responses_resp["usage"]["input_tokens"] == 5
    assert responses_resp["usage"]["output_tokens"] == 7


def test_streaming_end_to_end():
    converter = StreamConverter(response_id="resp-e2e", model="kimi-k2.6")

    events = [
        '{"choices":[{"delta":{"content":"Thinking"}}]}',
        '{"choices":[{"delta":{"content":"..."}}]}',
        '{"choices":[{"delta":{"content":" Done!"}}]}',
        "[DONE]",
    ]

    results = []
    for e in events:
        results.extend(converter.process_event(e))

    parsed = [json.loads(r) for r in results]
    types = [p["type"] for p in parsed]

    assert types == [
        "response.content_part.added",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.delta",
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
        "response.completed",
    ]

    assert parsed[1]["delta"] == "Thinking"
    assert parsed[2]["delta"] == "..."
    assert parsed[3]["delta"] == " Done!"
    assert parsed[4]["text"] == "Thinking... Done!"
