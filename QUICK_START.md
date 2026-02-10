# 快速参考：Deep Agents 旧版 OpenAI API 兼容性

## 问题症状

```
ValueError: content must be a string, not a list
ValueError: write_todos is not strict. Only strict function tools can be auto-parsed
```

## 快速修复

### 只需改一行代码

**之前**:
```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url="https://api.example.com/v1",
    api_key="your-key",
)
```

**之后**:
```python
from deepagents.chat_models.openai_compat import CompatibleChatOpenAI

model = CompatibleChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    base_url="https://api.example.com/v1",
    api_key="your-key",
)
```

## 完整示例

```python
from deepagents import create_deep_agent
from deepagents.chat_models.openai_compat import CompatibleChatOpenAI
from langchain_core.messages import HumanMessage

# 1. 创建兼容的模型
model = CompatibleChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key="your-api-key",
    base_url="https://api.example.com/v1",
    temperature=0.7,
)

# 2. 创建 agent
agent = create_deep_agent(model=model)

# 3. 使用 agent
result = agent.invoke(
    {"messages": [HumanMessage(content="你好")]},
    {"configurable": {"thread_id": "test"}},
)

print(result)
```

## 工作原理

`CompatibleChatOpenAI` 自动将消息内容从数组格式转换为字符串：

```
输入:  [{"type": "text", "text": "Hello"}]
输出:  "Hello"

输入:  [{"type": "text", "text": "A"}, {"type": "text", "text": "B"}]
输出:  "A\nB"
```

## 支持的模型

- ✅ DeepSeek
- ✅ 本地 LLM（OpenAI 兼容接口）
- ✅ 其他旧版 OpenAI API

## 常见问题

**Q: 我需要修改现有代码吗？**
A: 不需要。只需将 `ChatOpenAI` 改为 `CompatibleChatOpenAI`。

**Q: 这会影响性能吗？**
A: 不会。转换开销极小。

**Q: 支持多模态吗？**
A: 图片和文件会被转换为文本占位符。如果需要完整多模态支持，请使用新版 OpenAI API。

**Q: 可以同时使用两个模型吗？**
A: 可以。`CompatibleChatOpenAI` 和 `ChatOpenAI` 可以在同一个应用中使用。

## 更多信息

详见 `OPENAI_COMPAT_GUIDE.md`
