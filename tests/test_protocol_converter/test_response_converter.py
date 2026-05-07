from src.openai_protocol_converter.response_converter import convert_response


def test_basic_response_conversion():
    chat_resp = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello! How can I help?",
                },
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 7,
            "total_tokens": 17,
        },
    }
    result = convert_response(chat_resp)
    assert result["id"] == "chatcmpl-123"
    assert result["object"] == "response"
    assert result["created_at"] == 1700000000
    assert result["model"] == "kimi-k2.6"
    assert result["status"] == "completed"
    assert len(result["output"]) == 1
    assert result["output"][0]["type"] == "message"
    assert result["output"][0]["role"] == "assistant"
    assert len(result["output"][0]["content"]) == 1
    assert result["output"][0]["content"][0] == {
        "type": "output_text",
        "text": "Hello! How can I help?",
    }


def test_empty_content():
    chat_resp = {
        "id": "chatcmpl-456",
        "object": "chat.completion",
        "created": 1700000001,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": ""},
                "finish_reason": "stop",
            },
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 0, "total_tokens": 5},
    }
    result = convert_response(chat_resp)
    assert result["output"][0]["content"][0]["text"] == ""


def test_usage_mapping():
    chat_resp = {
        "id": "chatcmpl-789",
        "object": "chat.completion",
        "created": 1700000002,
        "model": "kimi-k2.6",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hi"},
                "finish_reason": "stop",
            },
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        },
    }
    result = convert_response(chat_resp)
    assert result["usage"]["input_tokens"] == 100
    assert result["usage"]["output_tokens"] == 50
    assert result["usage"]["total_tokens"] == 150
