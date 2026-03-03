"""OpenAI API 兼容性中间件，用于兼容仅支持字符串内容的旧版 API。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from langchain_core.messages import BaseMessage, SystemMessage
from typing_extensions import override

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langgraph.runtime import Runtime


def _flatten_content_to_string(content: Any) -> str:
    """将内容块列表转换为单一字符串，用于兼容旧版 OpenAI API。

    旧版 OpenAI API 要求消息的 content 字段为字符串，而新版 API 支持内容块数组。
    本函数将各种格式的内容统一转换为字符串。

    Args:
        content: 待转换的内容，可以是字符串或内容块列表。

    Returns:
        内容的单一字符串表示。
    """
    # 若已是字符串，直接返回
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                # 字符串块直接追加
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    # 文本块：提取 text 字段
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    # 图片块：用占位符替代
                    parts.append("[Image]")
                elif block.get("type") == "file":
                    # 文件块：用占位符替代
                    parts.append("[File]")
                else:
                    # 其他类型块：尝试提取 text 字段
                    if "text" in block:
                        parts.append(block["text"])
        # 过滤空字符串后用换行符拼接
        return "\n".join(filter(None, parts))

    # 其他类型直接转为字符串
    return str(content)


class OpenAICompatMiddleware(AgentMiddleware):
    """兼容旧版 OpenAI API 的中间件。

    将消息内容从数组格式（新版 OpenAI API 使用）转换为字符串格式
    （旧版 OpenAI API 要求 content 字段必须为字符串，不支持数组）。

    适用于使用旧版协议规范的 OpenAI 兼容 API。

    示例：
        ```python
        from deepagents import create_deep_agent
        from deepagents.middleware.openai_compat import OpenAICompatMiddleware
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model="deepseek-ai/DeepSeek-V3",
            base_url="https://api.example.com/v1",
        )

        agent = create_deep_agent(
            model=model,
            middleware=[OpenAICompatMiddleware()]
        )
        ```
    """

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """将消息内容转换为字符串格式，以兼容旧版 OpenAI API（同步版本）。

        若系统消息的 content 为列表（内容块数组），则将其展平为字符串。

        Args:
            request: 待执行的模型请求。
            handler: 执行模型请求的回调函数。

        Returns:
            模型调用结果。
        """
        # 若存在系统消息且其内容为列表格式，则转换为字符串
        if request.system_message is not None:
            system_content = request.system_message.content
            if isinstance(system_content, list):
                # 将内容块数组展平为字符串
                flattened = _flatten_content_to_string(system_content)
                request = request.override(
                    system_message=SystemMessage(content=flattened)
                )

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """将消息内容转换为字符串格式，以兼容旧版 OpenAI API（异步版本）。

        Args:
            request: 待执行的模型请求。
            handler: 异步执行模型请求的回调函数。

        Returns:
            模型调用结果。
        """
        # 若存在系统消息且其内容为列表格式，则转换为字符串
        if request.system_message is not None:
            system_content = request.system_message.content
            if isinstance(system_content, list):
                # 将内容块数组展平为字符串
                flattened = _flatten_content_to_string(system_content)
                request = request.override(
                    system_message=SystemMessage(content=flattened)
                )

        return await handler(request)
