import sys
sys.path.insert(0, "/var/work/kimi-open-responses/src")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "request_converter",
    "/var/work/kimi-open-responses/src/openai_protocol_converter/request_converter.py",
)
request_converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(request_converter)
convert_request = request_converter.convert_request


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
    assert result["max_tokens"] == 100
    assert result["top_p"] == 0.9
    assert result["presence_penalty"] == 0.5
    assert result["frequency_penalty"] == 0.3
    assert result["stream"] is True
