# OpenAI Protocol Converter — llm_router 集成技术文档

## 1. 概述

本文档面向 llm_router 开发团队，描述如何将 `openai_protocol_converter` 库集成到现有的 mitmproxy 路由代理中。

**核心目标：** 让 llm_router 能够接收 **OpenAI responses API**（新协议）格式的请求，通过转换库将其转为 **chat.completions**（旧协议）格式后转发给 Kimi 2.6 上游，并将上游返回的 chat.completions 响应转回 responses API 格式返回给客户端。

**转换方向：**
- 请求：`responses API → chat.completions`（由 `convert_request()` 处理）
- 响应：`chat.completions → responses API`（由 `convert_response()` / `StreamConverter` 处理）

---

## 2. 库结构

```
openai_protocol_converter/
├── __init__.py           # 懒加载导出：convert_request, convert_response, StreamConverter
├── request_converter.py  # convert_request(responses_req: dict) -> dict
├── response_converter.py # convert_response(chat_resp: dict) -> dict
└── stream_converter.py   # StreamConverter(response_id: str, model: str)
```

### 2.1 核心 API

#### `convert_request(responses_req: dict) -> dict`

将 responses API 请求体转换为 chat.completions 请求体。

**支持的字段映射：**

| responses API 字段 | chat.completions 字段 | 说明 |
|-------------------|---------------------|------|
| `input` (str) | `messages` | `[{"role": "user", "content": input}]` |
| `input` (list) | `messages` | 直接透传（已兼容 messages 格式） |
| `instructions` | `messages[0]` | 前置为 system message |
| `model` | `model` | 直接透传 |
| `temperature` | `temperature` | 直接透传 |
| `max_output_tokens` | `max_tokens` | 字段重命名 |
| `top_p` | `top_p` | 直接透传 |
| `presence_penalty` | `presence_penalty` | 直接透传 |
| `frequency_penalty` | `frequency_penalty` | 直接透传 |
| `tools` | `tools` | 直接透传（格式已兼容） |
| `tool_choice` | `tool_choice` | 直接透传 |
| `stream` | `stream` | 直接透传 |
| `text.format` | `response_format` | JSON Schema 格式直接透传 |
| `reasoning.effort` | `thinking.type` | `"none"` → `"disabled"`，其他 → `"enabled"` |

**不处理的字段（需由 llm_router 在外部处理）：**
- `previous_response_id` — 由 llm_router 查询历史消息后注入到 `input` 中

#### `convert_response(chat_resp: dict) -> dict`

将 chat.completions 响应体转换为 responses API 响应体（非流式）。

**字段映射：**

| chat.completions 字段 | responses API 字段 | 说明 |
|---------------------|-------------------|------|
| `choices[0].message.content` | `output[0].content` | 包装为 `output_text` |
| `choices[0].message.tool_calls` | `output[0].content` | 每个 tool_call → `output_function_call` |
| `choices[0].message.refusal` | `output[0].content` | 包装为 `refusal` |
| `choices[0].message.role` | `output[0].role` | 直接透传 |
| `usage.prompt_tokens` | `usage.input_tokens` | 重命名 |
| `usage.completion_tokens` | `usage.output_tokens` | 重命名 |
| `usage.total_tokens` | `usage.total_tokens` | 直接透传 |
| `id` | `id` | 直接透传 |
| `created` | `created_at` | 直接透传 |
| — | `status` | 固定为 `"completed"` |
| — | `object` | 固定为 `"response"` |

#### `StreamConverter(response_id: str, model: str)`

用于流式（SSE）响应的逐事件转换。

**方法：** `process_event(event_data: str) -> str | None`

- 输入：chat.completions SSE 事件 JSON 字符串
- 输出：responses API SSE 事件 JSON 字符串，或 `None`（跳过该事件）
- 遇到 `"[DONE]"` 时返回 `{"status": "completed"}`

**使用方式：**

```python
converter = StreamConverter(response_id="resp-123", model="kimi-k2.6")

for sse_event in upstream_stream:
    converted = converter.process_event(sse_event)
    if converted:
        yield f"data: {converted}\n\n"
```

---

## 3. 集成步骤

### 3.1 复制库文件

将 `src/openai_protocol_converter/` 目录复制到 llm_router 项目的合适位置，例如：

```
llm_router/
├── src/
│   ├── openai_protocol_converter/   # <-- 复制到此
│   │   ├── __init__.py
│   │   ├── request_converter.py
│   │   ├── response_converter.py
│   │   └── stream_converter.py
│   ├── proxy.py
│   ├── ...
```

### 3.2 导入方式

在 `proxy.py` 中导入：

```python
from src.openai_protocol_converter import convert_request, convert_response, StreamConverter
```

（如果复制到不同路径，调整导入语句）

---

## 4. 数据库变更

### 4.1 模型配置表增加协议版本字段

在 `model_configs` 表中增加 `protocol_version` 字段，用于标识该上游是否走协议转换：

```sql
ALTER TABLE model_configs ADD COLUMN protocol_version TEXT DEFAULT 'chat_completions';
```

**取值说明：**
- `'chat_completions'` — 现有行为，直接透传（默认）
- `'responses_api'` — 启用协议转换：请求用 `convert_request()`，响应用 `convert_response()` / `StreamConverter`

### 4.2 llm_calls 表增加 protocol_version（可选）

如需记录每次调用使用的协议版本：

```sql
ALTER TABLE llm_calls ADD COLUMN protocol_version TEXT DEFAULT 'chat_completions';
```

---

## 5. proxy.py 修改详解

### 5.1 请求处理流程（`request()` 方法）

在 `proxy.py` 的请求处理流程中，找到**模型匹配后的转发逻辑**，插入协议转换：

```python
# === 在 proxy.py 的 request() 方法中 ===
# 位置：匹配 model 映射后，构建上游请求前

mapping, is_default = self._match_model(model_name)
if mapping is None:
    # ... 现有 404 处理 ...
    return

# 检查是否需要协议转换
needs_conversion = mapping.get("protocol_version") == "responses_api"

if needs_conversion:
    # 1. 处理 previous_response_id（如果存在）
    body_dict = json.loads(captured_req.body)
    previous_id = body_dict.get("previous_response_id")
    if previous_id:
        # 查询历史调用记录（按 api_key_id 隔离）
        history = self.storage.get_call_history(previous_id, api_key_id)
        if history is None:
            flow.response = http.Response.make(
                400,
                json.dumps({
                    "error": {
                        "type": "invalid_request_error",
                        "code": "invalid_id",
                        "message": "Previous response not found"
                    }
                }).encode(),
                {"Content-Type": "application/json"}
            )
            return
        
        # 将历史消息注入到 input 中
        body_dict = self._inject_history_into_input(body_dict, history)
        captured_req.body = json.dumps(body_dict)
    
    # 2. 转换请求体
    converted_body = convert_request(body_dict)
    captured_req.body = json.dumps(converted_body, ensure_ascii=False)
    flow.request.content = captured_req.body.encode("utf-8")

# ... 继续原有转发逻辑 ...
```

#### `_inject_history_into_input()` 实现参考

```python
def _inject_history_into_input(self, body_dict: dict, history: dict) -> dict:
    """将历史调用的消息注入到当前请求的 input 中。
    
    history: 数据库中查到的 llm_calls 记录，包含 request_body 和 response_body
    """
    # 解析历史请求和响应
    prev_request = json.loads(history["request_body"])
    prev_response = json.loads(history["response_body"])
    
    # 构建历史 messages
    messages = []
    
    # 添加历史 input 中的消息
    prev_input = prev_request.get("input", "")
    if isinstance(prev_input, str):
        messages.append({"role": "user", "content": prev_input})
    elif isinstance(prev_input, list):
        messages.extend(prev_input)
    
    # 添加历史响应中的 assistant 消息
    # 从 responses API 格式提取 assistant 回复
    for output_item in prev_response.get("output", []):
        if output_item.get("type") == "message":
            content_parts = []
            for part in output_item.get("content", []):
                if part.get("type") == "output_text":
                    content_parts.append(part.get("text", ""))
            if content_parts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(content_parts)
                })
    
    # 将历史消息 + 当前 input 合并
    current_input = body_dict.get("input", "")
    if isinstance(current_input, str):
        messages.append({"role": "user", "content": current_input})
    elif isinstance(current_input, list):
        messages.extend(current_input)
    
    body_dict["input"] = messages
    # 移除 previous_response_id，因为已通过 input 注入
    body_dict.pop("previous_response_id", None)
    
    return body_dict
```

#### `storage.get_call_history()` 实现参考

在 `src/storage.py` 中新增方法：

```python
def get_call_history(self, call_id: str, api_key_id: int) -> dict | None:
    """查询历史调用记录，按 api_key_id 隔离。
    
    返回 llm_calls 记录（包含 request_body 和 response_body），
    如果找不到或不属于该 api_key_id，返回 None。
    """
    # SQLite 实现
    cur = self._conn.cursor()
    cur.execute(
        "SELECT request_body, response_body FROM llm_calls WHERE call_id = ? AND api_key_id = ?",
        (call_id, api_key_id)
    )
    row = cur.fetchone()
    cur.close()
    
    if row:
        return {"request_body": row[0], "response_body": row[1]}
    return None
```

### 5.2 响应处理流程（`response()` 方法）

在 `proxy.py` 的响应处理流程中，插入协议转换：

```python
# === 在 proxy.py 的 response() 方法中 ===
# 位置：获取 captured_req 后，保存到数据库前

captured_req = self._pop_pending_request(flow)
if captured_req is None:
    return

# 获取上游配置（需要从 flow metadata 或重新查询）
mapping = flow.metadata.get("model_mapping", {})
needs_conversion = mapping.get("protocol_version") == "responses_api"

if needs_conversion:
    # 判断是否为流式响应
    is_stream = self._is_stream_request(captured_req.body)
    
    if is_stream:
        # 流式响应：使用 StreamConverter 逐事件转换
        # 注意：流式转换需要在 responseheaders() hook 中设置
        # 这里只记录转换器状态供后续使用
        flow.metadata["stream_converter"] = StreamConverter(
            response_id=captured_req.call_id,
            model=captured_req.overridden_model or captured_req.original_model
        )
    else:
        # 非流式响应：直接转换响应体
        try:
            chat_resp = json.loads(captured_resp.body)
            responses_resp = convert_response(chat_resp)
            # 更新响应体
            new_body = json.dumps(responses_resp, ensure_ascii=False)
            flow.response.content = new_body.encode("utf-8")
            # 更新 captured_resp 以便正确记录
            captured_resp.body = new_body
            # 更新响应头 Content-Length
            flow.response.headers["Content-Length"] = str(len(new_body.encode("utf-8")))
        except Exception as e:
            logger.error(f"Response conversion failed: {e}")
            # 转换失败时透传原始响应，避免破坏客户端体验
```

### 5.3 流式响应处理（`responseheaders()` 方法）

对于流式响应，需要在 `responseheaders()` hook 中拦截 SSE 流并进行实时转换：

```python
def responseheaders(self, flow: http.HTTPFlow):
    """响应头到达时触发"""
    if flow.metadata.get("local_response"):
        return
    
    flow.metadata["headers_time"] = time.time()
    
    # 检查是否需要流式协议转换
    mapping = flow.metadata.get("model_mapping", {})
    needs_conversion = mapping.get("protocol_version") == "responses_api"
    is_stream = self._is_stream_request(flow.metadata.get("request_body_for_stream"))
    
    if needs_conversion and is_stream:
        # 创建 StreamConverter 并设置为流处理器
        converter = StreamConverter(
            response_id=flow.metadata.get("call_id", str(uuid.uuid4())),
            model=flow.metadata.get("overridden_model", flow.metadata.get("original_model", "unknown"))
        )
        flow.metadata["protocol_converter"] = converter
        
        # 包装原始 stream 处理器，添加转换层
        original_stream = flow.response.stream
        
        def converted_stream(chunk: bytes) -> bytes:
            # 解析 SSE 事件
            event_text = chunk.decode("utf-8", errors="replace")
            
            # 处理每个 data: 行
            lines = event_text.strip().split("\n")
            output_lines = []
            
            for line in lines:
                if line.startswith("data: "):
                    data = line[6:]
                    converted = converter.process_event(data)
                    if converted:
                        output_lines.append(f"data: {converted}")
                else:
                    output_lines.append(line)
            
            if output_lines:
                return ("\n".join(output_lines) + "\n\n").encode("utf-8")
            return b""
        
        flow.response.stream = converted_stream
```

> **注意：** 上述流式处理是简化版本。实际集成时可能需要更精细的 SSE 解析，因为 SSE 事件可能跨多个 chunk 到达。建议参考 mitmproxy 的流式文档和现有的 `_capture_stream_chunk` 实现。

### 5.4 存储模型映射到 flow metadata

在 `request()` 方法中匹配到模型映射后，将映射信息保存到 flow.metadata，供响应阶段使用：

```python
# 在 request() 方法中，匹配 model 后
mapping, is_default = self._match_model(model_name)
# ...
flow.metadata["model_mapping"] = mapping
flow.metadata["original_model"] = model_name
flow.metadata["overridden_model"] = mapping.get("forward_model", model_name)
```

---

## 6. previous_response_id 详细设计

### 6.1 数据流

```
客户端发送请求
├── 包含 previous_response_id: "resp-abc"
│
llm_router
├── 验证 API Key → 获取 api_key_id
├── 查询数据库: SELECT * FROM llm_calls WHERE call_id="resp-abc" AND api_key_id=?
├── 如果找不到 → 返回 400 invalid_id
├── 如果找到 → 提取历史 request_body + response_body
├── 将历史对话注入到当前 input 中
├── 移除 previous_response_id 字段
├── 调用 convert_request() 转换
└── 转发给上游

上游返回响应
├── llm_router 调用 convert_response() 转换（如需要）
├── 保存到数据库（包含新的 call_id 作为 response_id）
└── 返回给客户端（客户端收到新的 response_id）
```

### 6.2 隔离策略

- 按 **API key** 级别隔离
- 查询条件必须包含 `AND api_key_id = ?`
- 如果 `previous_response_id` 属于其他 API key，返回与"找不到"相同的错误（防止信息泄露）

### 6.3 response_id 生成

- `call_id`（数据库主键）同时充当 `response_id`
- 必须是 UUID 格式，与 OpenAI 行为一致
- 在请求处理阶段生成（在调用 convert_request 之前），以便后续轮次可以引用

### 6.4 历史消息注入算法

```python
def build_messages_from_history(prev_request, prev_response, current_input):
    """从历史调用构建 messages 数组。"""
    messages = []
    
    # 1. 历史请求中的 input → user message(s)
    prev_input = prev_request.get("input", "")
    if isinstance(prev_input, str):
        messages.append({"role": "user", "content": prev_input})
    elif isinstance(prev_input, list):
        messages.extend(prev_input)
    
    # 2. 历史响应中的 output → assistant message
    # responses API 格式：output[0].content[...].text
    for item in prev_response.get("output", []):
        if item.get("type") == "message":
            texts = []
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    texts.append(part["text"])
            if texts:
                messages.append({
                    "role": "assistant",
                    "content": "\n".join(texts)
                })
    
    # 3. 当前 input → user message(s)
    if isinstance(current_input, str):
        messages.append({"role": "user", "content": current_input})
    elif isinstance(current_input, list):
        messages.extend(current_input)
    
    return messages
```

---

## 7. 配置说明

### 7.1 数据库配置示例

在 llm_router 的 Web 控制台或管理 API 中，为模型配置增加 `protocol_version` 字段：

```json
{
  "model_key": "kimi-k2.6",
  "target_base_url": "https://api.moonshot.cn/v1",
  "api_key": "sk-...",
  "forward_model": "kimi-k2.6",
  "protocol_version": "responses_api",
  "is_active": true,
  "is_default": false
}
```

### 7.2 向后兼容

- 现有配置不设置 `protocol_version` 时，默认值为 `'chat_completions'`，保持现有行为不变
- 只有在显式设置为 `'responses_api'` 时才启用协议转换

---

## 8. 错误处理

### 8.1 转换器层面的错误

转换器本身是无状态纯函数，**不处理异常输入**。如果输入格式不正确，会抛出标准 Python 异常（`KeyError`, `TypeError` 等）。

llm_router 应该在调用转换器前验证输入，并在转换失败时返回适当的 HTTP 错误：

```python
try:
    converted = convert_request(body_dict)
except (KeyError, TypeError, ValueError) as e:
    logger.warning(f"Request conversion failed: {e}")
    flow.response = http.Response.make(
        400,
        json.dumps({
            "error": {
                "type": "invalid_request_error",
                "message": f"Invalid request format: {e}"
            }
        }).encode(),
        {"Content-Type": "application/json"}
    )
    return
```

### 8.2 previous_response_id 错误

| 场景 | HTTP 状态码 | 响应体 |
|------|-----------|--------|
| previous_response_id 不存在 | 400 | `{"error": {"type": "invalid_request_error", "code": "invalid_id"}}` |
| previous_response_id 属于其他 API key | 400 | 同上（不泄露存在性信息） |
| 历史消息解析失败 | 500 | `{"error": {"type": "internal_error"}}` |

---

## 9. 测试建议

### 9.1 llm_router 层面的集成测试

建议在 llm_router 的测试套件中增加以下测试：

```python
def test_responses_api_request_conversion():
    """测试 responses API 请求走协议转换后正确转发"""
    # 1. 配置 model_config 的 protocol_version = "responses_api"
    # 2. 发送 responses API 格式请求
    # 3. 验证上游收到的是 chat.completions 格式
    # 4. 验证响应被转回 responses API 格式


def test_previous_response_id_chain():
    """测试多轮对话 previous_response_id 链式引用"""
    # 1. 第一轮：发送请求，获取 response_id
    # 2. 第二轮：使用 previous_response_id 发送新请求
    # 3. 验证上游收到的 messages 包含历史对话
    # 4. 验证第二轮的 response_id 也能被第三轮引用


def test_previous_response_id_isolation():
    """测试 previous_response_id 按 API key 隔离"""
    # 1. 用 API key A 发送请求，获取 response_id
    # 2. 用 API key B 尝试引用该 response_id
    # 3. 验证返回 400 invalid_id


def test_responses_api_streaming():
    """测试 responses API 流式响应转换"""
    # 1. 发送 stream=true 的 responses API 请求
    # 2. 验证 SSE 事件格式为 responses API 格式
    # 3. 验证最后一个事件是 status: completed
```

### 9.2 与现有功能的兼容性

- 确保协议转换不影响现有的 API key 验证、调用记录、多上游路由等功能
- 确保 `use_claude_features` 和 `use_roo_features` 在协议转换模式下仍然有效
- 确保健康检查不受协议转换影响（健康检查使用 chat.completions 格式）

---

## 10. 常见问题

### Q: 为什么 previous_response_id 不由转换器内部处理？

A: 转换器是无状态纯函数库，不涉及数据库查询或外部状态。`previous_response_id` 需要查询历史调用记录，这是有状态操作，由 llm_router 负责。

### Q: 流式转换会引入延迟吗？

A: 转换是纯内存操作（JSON 解析 + 字段映射），延迟在微秒级别，对 SSE 流的实时性影响可忽略。

### Q: 支持多 choices 吗？

A: 当前实现只处理 `choices[0]`，与 OpenAI 和 Kimi 的默认行为一致。如需支持多 choices，需要扩展转换器。

### Q: 工具调用结果（tool output）如何传递？

A: 工具调用结果是客户端行为。客户端收到 `output_function_call` 后，应调用对应函数，然后将结果作为新的 `input` 发送（包含 `previous_response_id`）。转换器只负责格式转换，不处理业务逻辑。

---

## 11. 附录：完整数据示例

### 请求转换示例

**responses API 请求（客户端发送）：**
```json
{
  "model": "kimi-k2.6",
  "input": "What is the weather?",
  "temperature": 0.5,
  "reasoning": {"effort": "medium"}
}
```

**转换后（转发给 Kimi）：**
```json
{
  "model": "kimi-k2.6",
  "messages": [{"role": "user", "content": "What is the weather?"}],
  "temperature": 0.5,
  "thinking": {"type": "enabled"}
}
```

### 响应转换示例

**chat.completions 响应（Kimi 返回）：**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1700000000,
  "model": "kimi-k2.6",
  "choices": [{
    "index": 0,
    "message": {"role": "assistant", "content": "I don't know."},
    "finish_reason": "stop"
  }],
  "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13}
}
```

**转换后（返回给客户端）：**
```json
{
  "id": "chatcmpl-123",
  "object": "response",
  "created_at": 1700000000,
  "model": "kimi-k2.6",
  "output": [{
    "type": "message",
    "role": "assistant",
    "content": [{"type": "output_text", "text": "I don't know."}]
  }],
  "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
  "status": "completed"
}
```
