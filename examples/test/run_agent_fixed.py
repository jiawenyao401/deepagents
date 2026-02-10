from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage
from typing import Any, List

class StringContentChatOpenAI(ChatOpenAI):
    """强制content为字符串的ChatOpenAI包装器"""
    
    def _convert_messages_to_dicts(self, messages: List[BaseMessage]) -> List[dict]:
        """重写消息转换，确保content始终是字符串"""
        dicts = super()._convert_messages_to_dicts(messages)
        for msg in dicts:
            # 如果content是列表，提取文本内容
            if isinstance(msg.get("content"), list):
                text_parts = []
                for part in msg["content"]:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        text_parts.append(part)
                msg["content"] = " ".join(text_parts)
        return dicts

print("配置模型...")
model = StringContentChatOpenAI(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJndE00Y1dCVmlpRWVFTmIySzJrUU1wemFTYlVxNDVGRSJ9._ATcBvOhS5fZzkGeCrjAUlOWNFYil2KIAo7mlcPReUE",
    base_url="https://ai-dcin-test.digitalyili.com/v1"
)

print("创建agent...")
agent = create_deep_agent(model=model)

print("调用agent...")
result = agent.invoke({"messages": [{"role": "user", "content": "你好，请介绍一下自己"}]})
print("\n结果:", result)
