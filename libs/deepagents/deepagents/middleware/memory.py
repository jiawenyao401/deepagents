# ruff: noqa: E501  # 系统提示字符串较长，禁用行长检查
"""从 AGENTS.md 文件加载 Agent 记忆/上下文的中间件。

本模块实现了对 AGENTS.md 规范（https://agents.md/）的支持：
从可配置的来源加载记忆/上下文，并将其注入系统提示。

## 概述

AGENTS.md 文件提供项目特定的上下文和指令，帮助 AI Agent 高效工作。
与技能（按需工作流）不同，记忆始终被加载，提供持久化上下文。

## 使用示例

```python
from deepagents import MemoryMiddleware
from deepagents.backends.filesystem import FilesystemBackend

# 安全提示：FilesystemBackend 允许读写整个文件系统。
# 请确保 Agent 运行在沙箱中，或为文件操作添加人工审批（HIL）。
backend = FilesystemBackend(root_dir="/")

middleware = MemoryMiddleware(
    backend=backend,
    sources=[
        "~/.deepagents/AGENTS.md",
        "./.deepagents/AGENTS.md",
    ],
)

agent = create_deep_agent(middleware=[middleware])
```

## 记忆来源

来源是指向 AGENTS.md 文件的路径列表，按顺序加载并合并。
多个来源按顺序拼接，后面的来源内容出现在前面来源之后。

## 文件格式

AGENTS.md 文件是标准 Markdown，无固定结构要求。
常见章节包括：
- 项目概述
- 构建/测试命令
- 代码风格指南
- 架构说明
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Annotated, NotRequired, TypedDict

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
    ResponseT,
)
from langchain.tools import ToolRuntime

from deepagents.middleware._utils import append_to_system_message

# 模块级日志记录器
logger = logging.getLogger(__name__)


class MemoryState(AgentState):
    """MemoryMiddleware 的状态模式。

    Attributes:
        memory_contents: 将来源路径映射到其加载内容的字典。
            标记为私有，不会包含在最终的 Agent 状态中。
    """

    # 私有字段：存储各来源路径对应的文件内容，不对外暴露
    memory_contents: NotRequired[Annotated[dict[str, str], PrivateStateAttr]]


class MemoryStateUpdate(TypedDict):
    """MemoryMiddleware 的状态更新结构。"""

    # 更新后的记忆内容字典（路径 -> 内容）
    memory_contents: dict[str, str]


# 记忆系统提示模板：将加载的记忆内容和使用指南注入系统提示
MEMORY_SYSTEM_PROMPT = """<agent_memory>
{agent_memory}
</agent_memory>

<memory_guidelines>
    The above <agent_memory> was loaded in from files in your filesystem. As you learn from your interactions with the user, you can save new knowledge by calling the `edit_file` tool.

    **Learning from feedback:**
    - One of your MAIN PRIORITIES is to learn from your interactions with the user. These learnings can be implicit or explicit. This means that in the future, you will remember this important information.
    - When you need to remember something, updating memory must be your FIRST, IMMEDIATE action - before responding to the user, before calling other tools, before doing anything else. Just update memory immediately.
    - When user says something is better/worse, capture WHY and encode it as a pattern.
    - Each correction is a chance to improve permanently - don't just fix the immediate issue, update your instructions.
    - A great opportunity to update your memories is when the user interrupts a tool call and provides feedback. You should update your memories immediately before revising the tool call.
    - Look for the underlying principle behind corrections, not just the specific mistake.
    - The user might not explicitly ask you to remember something, but if they provide information that is useful for future use, you should update your memories immediately.

    **Asking for information:**
    - If you lack context to perform an action (e.g. send a Slack DM, requires a user ID/email) you should explicitly ask the user for this information.
    - It is preferred for you to ask for information, don't assume anything that you do not know!
    - When the user provides information that is useful for future use, you should update your memories immediately.

    **When to update memories:**
    - When the user explicitly asks you to remember something (e.g., "remember my email", "save this preference")
    - When the user describes your role or how you should behave (e.g., "you are a web researcher", "always do X")
    - When the user gives feedback on your work - capture what was wrong and how to improve
    - When the user provides information required for tool use (e.g., slack channel ID, email addresses)
    - When the user provides context useful for future tasks, such as how to use tools, or which actions to take in a particular situation
    - When you discover new patterns or preferences (coding styles, conventions, workflows)

    **When to NOT update memories:**
    - When the information is temporary or transient (e.g., "I'm running late", "I'm on my phone right now")
    - When the information is a one-time task request (e.g., "Find me a recipe", "What's 25 * 4?")
    - When the information is a simple question that doesn't reveal lasting preferences (e.g., "What day is it?", "Can you explain X?")
    - When the information is an acknowledgment or small talk (e.g., "Sounds good!", "Hello", "Thanks for that")
    - When the information is stale or irrelevant in future conversations
    - Never store API keys, access tokens, passwords, or any other credentials in any file, memory, or system prompt.
    - If the user asks where to put API keys or provides an API key, do NOT echo or save it.

    **Examples:**
    Example 1 (remembering user information):
    User: Can you connect to my google account?
    Agent: Sure, I'll connect to your google account, what's your google account email?
    User: john@example.com
    Agent: Let me save this to my memory.
    Tool Call: edit_file(...) -> remembers that the user's google account email is john@example.com

    Example 2 (remembering implicit user preferences):
    User: Can you write me an example for creating a deep agent in LangChain?
    Agent: Sure, I'll write you an example for creating a deep agent in LangChain <example code in Python>
    User: Can you do this in JavaScript
    Agent: Let me save this to my memory.
    Tool Call: edit_file(...) -> remembers that the user prefers to get LangChain code examples in JavaScript
    Agent: Sure, here is the JavaScript example<example code in JavaScript>

    Example 3 (do not remember transient information):
    User: I'm going to play basketball tonight so I will be offline for a few hours.
    Agent: Okay I'll add a block to your calendar.
    Tool Call: create_calendar_event(...) -> just calls a tool, does not commit anything to memory, as it is transient information
</memory_guidelines>
"""


class MemoryMiddleware(AgentMiddleware[MemoryState, ContextT, ResponseT]):
    """从 AGENTS.md 文件加载 Agent 记忆的中间件。

    从配置的来源加载记忆内容，并将其注入系统提示。
    支持多个来源，所有来源内容会被合并。

    Args:
        backend: 用于文件操作的后端实例或工厂函数。
        sources: 指定路径和名称的 MemorySource 配置列表。
    """

    # 声明该中间件使用的状态模式
    state_schema = MemoryState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        sources: list[str],
    ) -> None:
        """初始化记忆中间件。

        Args:
            backend: 后端实例或工厂函数（接收 runtime 并返回后端）。
                     对于 StateBackend，请使用工厂函数。
            sources: 要加载的记忆文件路径列表（如 `["~/.deepagents/AGENTS.md",
                     "./.deepagents/AGENTS.md"]`）。

                     显示名称会自动从路径中派生。

                     来源按顺序加载。
        """
        self._backend = backend  # 存储后端实例或工厂函数
        self.sources = sources   # 存储记忆文件路径列表

    def _get_backend(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """从实例或工厂函数解析后端。

        若 _backend 是可调用对象（工厂函数），则构造一个临时的 ToolRuntime
        来调用工厂函数获取后端实例；否则直接返回后端实例。

        Args:
            state: 当前 Agent 状态。
            runtime: 工厂函数所需的运行时上下文。
            config: 传递给后端工厂的可运行配置。

        Returns:
            解析后的后端实例。
        """
        if callable(self._backend):
            # 构造一个临时的 ToolRuntime，用于调用后端工厂函数
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]
        # 直接返回后端实例
        return self._backend

    def _format_agent_memory(self, contents: dict[str, str]) -> str:
        """将记忆内容格式化为带路径标注的字符串。

        将各来源路径与对应内容配对，拼接成完整的记忆提示文本。
        若无任何内容，则返回"无记忆已加载"的提示。

        Args:
            contents: 来源路径到内容的映射字典。

        Returns:
            包含路径+内容对的格式化字符串，外层用 <agent_memory> 标签包裹。
        """
        # 若内容字典为空，返回无记忆提示
        if not contents:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

        # 按 sources 顺序拼接各来源的路径和内容
        sections = [f"{path}\n{contents[path]}" for path in self.sources if contents.get(path)]

        # 若所有来源均无内容，返回无记忆提示
        if not sections:
            return MEMORY_SYSTEM_PROMPT.format(agent_memory="(No memory loaded)")

        # 用双换行符分隔各来源内容段落
        memory_body = "\n\n".join(sections)
        return MEMORY_SYSTEM_PROMPT.format(agent_memory=memory_body)

    def before_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:  # ty: ignore[invalid-method-override]
        """在 Agent 执行前同步加载记忆内容。

        从所有配置的来源加载记忆并存入状态。
        若状态中已存在记忆内容，则跳过加载（避免重复加载）。

        Args:
            state: 当前 Agent 状态。
            runtime: 运行时上下文。
            config: 可运行配置。

        Returns:
            包含已填充 memory_contents 的状态更新，若已存在则返回 None。
        """
        # 若状态中已有记忆内容，跳过加载
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        # 批量下载所有来源文件
        results = backend.download_files(list(self.sources))
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                # 文件不存在时跳过，其他错误则抛出异常
                if response.error == "file_not_found":
                    continue
                msg = f"Failed to download {path}: {response.error}"
                raise ValueError(msg)
            if response.content is not None:
                # 将文件内容解码为 UTF-8 字符串并存入字典
                contents[path] = response.content.decode("utf-8")
                logger.debug("Loaded memory from: %s", path)

        return MemoryStateUpdate(memory_contents=contents)

    async def abefore_agent(self, state: MemoryState, runtime: Runtime, config: RunnableConfig) -> MemoryStateUpdate | None:  # ty: ignore[invalid-method-override]
        """在 Agent 执行前异步加载记忆内容。

        从所有配置的来源加载记忆并存入状态。
        若状态中已存在记忆内容，则跳过加载（避免重复加载）。

        Args:
            state: 当前 Agent 状态。
            runtime: 运行时上下文。
            config: 可运行配置。

        Returns:
            包含已填充 memory_contents 的状态更新，若已存在则返回 None。
        """
        # 若状态中已有记忆内容，跳过加载
        if "memory_contents" in state:
            return None

        backend = self._get_backend(state, runtime, config)
        contents: dict[str, str] = {}

        # 异步批量下载所有来源文件
        results = await backend.adownload_files(list(self.sources))
        for path, response in zip(self.sources, results, strict=True):
            if response.error is not None:
                # 文件不存在时跳过，其他错误则抛出异常
                if response.error == "file_not_found":
                    continue
                msg = f"Failed to download {path}: {response.error}"
                raise ValueError(msg)
            if response.content is not None:
                # 将文件内容解码为 UTF-8 字符串并存入字典
                contents[path] = response.content.decode("utf-8")
                logger.debug("Loaded memory from: %s", path)

        return MemoryStateUpdate(memory_contents=contents)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """将记忆内容注入系统消息。

        从请求状态中读取已加载的记忆内容，格式化后追加到系统消息末尾。

        Args:
            request: 待修改的模型请求。

        Returns:
            注入了记忆内容的新模型请求。
        """
        # 从状态中获取记忆内容，若不存在则使用空字典
        contents = request.state.get("memory_contents", {})
        # 格式化记忆内容为系统提示文本
        agent_memory = self._format_agent_memory(contents)
        # 将记忆内容追加到系统消息
        new_system_message = append_to_system_message(request.system_message, agent_memory)
        return request.override(system_message=new_system_message)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """包装模型调用，将记忆内容注入系统提示（同步版本）。

        Args:
            request: 正在处理的模型请求。
            handler: 使用修改后请求调用的处理函数。

        Returns:
            处理函数返回的模型响应。
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """包装模型调用，将记忆内容注入系统提示（异步版本）。

        Args:
            request: 正在处理的模型请求。
            handler: 使用修改后请求调用的异步处理函数。

        Returns:
            处理函数返回的模型响应。
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)
