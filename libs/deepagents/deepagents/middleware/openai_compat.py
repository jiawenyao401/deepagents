"""Middleware for OpenAI API compatibility with older versions that only support string content."""

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
    """Convert content blocks to a single string for older OpenAI APIs.
    
    Args:
        content: The content to flatten (can be string or list of blocks)
        
    Returns:
        A single string representation of the content
    """
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image_url":
                    parts.append("[Image]")
                elif block.get("type") == "file":
                    parts.append("[File]")
                else:
                    # For other block types, try to extract text
                    if "text" in block:
                        parts.append(block["text"])
        return "\n".join(filter(None, parts))
    
    return str(content)


class OpenAICompatMiddleware(AgentMiddleware):
    """Middleware for compatibility with older OpenAI API versions.
    
    This middleware converts message content from array format (used by newer OpenAI APIs)
    to string format (required by older OpenAI APIs that don't support content arrays).
    
    This is useful when using OpenAI-compatible APIs that follow the older protocol
    specification where the `content` field must be a string, not an array.
    
    Example:
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
        """Convert message content to string format for older OpenAI APIs.
        
        Args:
            request: Model request to execute
            handler: Callback that executes the model request
            
        Returns:
            The model call result
        """
        # Convert system message if needed
        if request.system_message is not None:
            system_content = request.system_message.content
            if isinstance(system_content, list):
                # Flatten content blocks to string
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
        """Async version of wrap_model_call.
        
        Args:
            request: Model request to execute
            handler: Async callback that executes the model request
            
        Returns:
            The model call result
        """
        # Convert system message if needed
        if request.system_message is not None:
            system_content = request.system_message.content
            if isinstance(system_content, list):
                # Flatten content blocks to string
                flattened = _flatten_content_to_string(system_content)
                request = request.override(
                    system_message=SystemMessage(content=flattened)
                )
        
        return await handler(request)
