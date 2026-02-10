"""ChatOpenAI wrapper for compatibility with older API versions that only support string content."""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI as BaseChatOpenAI


def _flatten_content_to_string(content: Any) -> str:
    """Convert content blocks to a single string for older OpenAI APIs."""
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
                    if "text" in block:
                        parts.append(block["text"])
        return "\n".join(filter(None, parts))
    
    return str(content)


class CompatibleChatOpenAI(BaseChatOpenAI):
    """ChatOpenAI wrapper for compatibility with older OpenAI API versions.
    
    Converts message content from array format to string format and removes
    strict mode from tool calls for compatibility with older APIs.
    """

    def _convert_input(self, input_: LanguageModelInput) -> Any:
        """Convert input and flatten message content to strings."""
        converted = super()._convert_input(input_)
        messages = converted.to_messages()
        flattened_messages = []
        
        for msg in messages:
            if isinstance(msg, SystemMessage):
                flattened_content = _flatten_content_to_string(msg.content)
                flattened_messages.append(
                    SystemMessage(
                        content=flattened_content,
                        name=msg.name,
                        id=msg.id,
                        additional_kwargs=msg.additional_kwargs,
                    )
                )
            else:
                if isinstance(msg.content, list):
                    flattened_content = _flatten_content_to_string(msg.content)
                    msg_dict = msg.model_dump()
                    msg_dict["content"] = flattened_content
                    msg_class = type(msg)
                    flattened_messages.append(msg_class(**msg_dict))
                else:
                    flattened_messages.append(msg)
        
        from langchain_core.prompt_values import ChatPromptValue
        return ChatPromptValue(messages=flattened_messages)
    
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict:
        """Get request payload and disable strict mode for tool calls."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        
        # Remove strict mode from tools for compatibility with older APIs
        if "tools" in payload and isinstance(payload["tools"], list):
            for tool in payload["tools"]:
                if isinstance(tool, dict) and "function" in tool:
                    tool["function"].pop("strict", None)
        
        return payload
