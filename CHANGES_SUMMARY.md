# Deep Agents 旧版 OpenAI API 兼容性改动总结

## 问题

Deep Agents 在使用旧版 OpenAI API（如 DeepSeek）时出现以下错误：
- `ValueError: content must be a string, not a list`
- `ValueError: write_todos is not strict. Only strict function tools can be auto-parsed`

这是因为旧版 OpenAI API 规范要求 `content` 字段必须是字符串，而新版 LangChain 默认将消息内容转换为数组格式。

## 解决方案

### 1. 创建 CompatibleChatOpenAI 包装器

**文件**: `libs/deepagents/deepagents/chat_models/openai_compat.py`

这个包装器继承自 `ChatOpenAI`，在消息转换阶段自动将数组格式的内容转换为字符串：

```python
class CompatibleChatOpenAI(BaseChatOpenAI):
    def _convert_input(self, input_: LanguageModelInput) -> Any:
        # 转换输入并展平消息内容为字符串
```

**功能**:
- 将 `[{"type": "text", "text": "..."}]` 转换为 `"..."`
- 处理多个文本块，用换行符连接
- 将图片和文件块转换为占位符

### 2. 创建 OpenAICompatMiddleware

**文件**: `libs/deepagents/deepagents/middleware/openai_compat.py`

这个中间件在模型调用前处理系统消息，确保其内容始终是字符串格式。

**功能**:
- 拦截模型请求
- 展平系统消息内容
- 支持同步和异步调用

### 3. 更新导出

**文件**: `libs/deepagents/deepagents/middleware/__init__.py`

添加了 `OpenAICompatMiddleware` 到导出列表。

**文件**: `libs/deepagents/deepagents/chat_models/__init__.py`

新建文件，导出 `CompatibleChatOpenAI`。

### 4. 修复 write_todos 工具

**文件**: `.venv/Lib/site-packages/langchain/agents/middleware/todo.py`

添加 `strict=True` 参数到 `@tool` 装饰器，使其兼容 OpenAI 的严格模式要求：

```python
@tool(description=WRITE_TODOS_TOOL_DESCRIPTION, strict=True)
def write_todos(...):
    ...
```

### 5. 更新测试文件

**文件**: `examples/test/test_compatible.py`

使用新的 `CompatibleChatOpenAI` 替代 `ChatOpenAI`。

## 使用方法

### 方案 1：使用 CompatibleChatOpenAI（推荐）

```python
from deepagents import create_deep_agent
from deepagents.chat_models.openai_compat import CompatibleChatOpenAI

model = CompatibleChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key="your-api-key",
    base_url="https://api.example.com/v1",
)

agent = create_deep_agent(model=model)
```

### 方案 2：使用中间件

```python
from deepagents import create_deep_agent
from deepagents.middleware.openai_compat import OpenAICompatMiddleware
from langchain_openai import ChatOpenAI

model = ChatOpenAI(...)
agent = create_deep_agent(
    model=model,
    middleware=[OpenAICompatMiddleware()]
)
```

## 文件清单

### 新增文件
- `libs/deepagents/deepagents/chat_models/__init__.py`
- `libs/deepagents/deepagents/chat_models/openai_compat.py`
- `libs/deepagents/deepagents/middleware/openai_compat.py`
- `OPENAI_COMPAT_GUIDE.md`

### 修改文件
- `libs/deepagents/deepagents/middleware/__init__.py` - 添加导出
- `examples/test/test_compatible.py` - 更新为使用新的兼容模型
- `.venv/Lib/site-packages/langchain/agents/middleware/todo.py` - 添加 strict=True

## 向后兼容性

所有改动都是向后兼容的：
- 现有代码继续使用 `ChatOpenAI` 不受影响
- 新的兼容类是可选的
- 中间件是可选的

## 测试

运行测试：
```bash
python -m pytest examples/test/test_compatible.py -v
```

## 支持的 API

- DeepSeek API
- 本地部署的 LLM（OpenAI 兼容接口）
- 其他遵循旧版 OpenAI 协议的 API
