# Deep Agents 旧版 OpenAI API 兼容性指南

## 问题描述

某些 OpenAI 兼容的 API（如 DeepSeek、本地部署的模型等）使用旧版 OpenAI 协议规范，其中 `content` 字段只能是字符串类型，不支持数组格式。

当使用这些 API 时，Deep Agents 会遇到以下错误：
```
ValueError: `write_todos` is not strict. Only `strict` function tools can be auto-parsed
```

或者：
```
Error: content must be a string, not a list
```

## 解决方案

Deep Agents 提供了两种方式来支持旧版 OpenAI API：

### 方案 1：使用 CompatibleChatOpenAI（推荐）

使用 `CompatibleChatOpenAI` 包装器自动将消息内容从数组格式转换为字符串格式：

```python
from deepagents import create_deep_agent
from deepagents.chat_models.openai_compat import CompatibleChatOpenAI

# 配置兼容的模型
model = CompatibleChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key="your-api-key",
    base_url="https://api.example.com/v1",
    temperature=0.7,
)

# 创建 agent
agent = create_deep_agent(model=model)

# 使用 agent
result = agent.invoke(
    {"messages": [{"role": "user", "content": "你好，请介绍一下自己"}]},
    {"configurable": {"thread_id": "test-thread"}},
)
print(result)
```

### 方案 2：使用 OpenAICompatMiddleware

如果你已经有一个 ChatOpenAI 实例，可以使用中间件来处理兼容性：

```python
from deepagents import create_deep_agent
from deepagents.middleware.openai_compat import OpenAICompatMiddleware
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key="your-api-key",
    base_url="https://api.example.com/v1",
)

agent = create_deep_agent(
    model=model,
    middleware=[OpenAICompatMiddleware()]
)
```

## 工作原理

### CompatibleChatOpenAI

这个包装器在消息转换阶段拦截消息内容，将所有数组格式的内容块转换为单一字符串：

- `[{"type": "text", "text": "Hello"}]` → `"Hello"`
- `[{"type": "text", "text": "Hello"}, {"type": "text", "text": "World"}]` → `"Hello\nWorld"`
- 图片和文件块会被转换为占位符：`[Image]`、`[File]`

### OpenAICompatMiddleware

这个中间件在模型调用前处理系统消息，确保系统消息内容始终是字符串格式。

## 支持的 API

这个解决方案已测试支持：

- DeepSeek API
- 本地部署的 LLM（使用 OpenAI 兼容接口）
- 其他遵循旧版 OpenAI 协议的 API

## 限制

- 多模态内容（图片、文件）会被转换为文本占位符
- 如果你需要完整的多模态支持，请使用支持新版 OpenAI API 的模型

## 故障排除

### 仍然收到 "content must be a string" 错误

确保你使用的是 `CompatibleChatOpenAI` 而不是 `ChatOpenAI`：

```python
# ❌ 错误
from langchain_openai import ChatOpenAI
model = ChatOpenAI(...)

# ✅ 正确
from deepagents.chat_models.openai_compat import CompatibleChatOpenAI
model = CompatibleChatOpenAI(...)
```

### 工具调用失败

如果遇到工具相关的错误，确保：

1. 你的 API 支持函数调用（tool_calls）
2. 如果 API 不支持严格模式，可以在模型配置中禁用它

```python
model = CompatibleChatOpenAI(
    model="your-model",
    api_key="your-key",
    base_url="your-base-url",
    model_kwargs={"strict": False},  # 禁用严格模式
)
```

## 示例

完整的工作示例见 `examples/test/test_compatible.py`
