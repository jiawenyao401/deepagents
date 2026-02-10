from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from typing import List, Any, Optional
from langchain_core.outputs import ChatResult, ChatGeneration

class StringOnlyChatOpenAI(ChatOpenAI):
    """强制所有消息content为字符串"""
    
    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None
    ) -> tuple[list[dict], dict]:
        """重写消息创建，强制content为字符串"""
        params = self._client_params
        message_dicts = []
        
        for m in messages:
            content = m.content
            # 处理多模态content（列表格式）
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        text_parts.append(item)
                content = " ".join(text_parts) if text_parts else ""
            
            message_dict = {"role": m.type, "content": str(content)}
            message_dicts.append(message_dict)
        
        return message_dicts, params

print("配置模型...")
model = StringOnlyChatOpenAI(
    model="Qwen/Qwen2.5-VL-32B-Instruct",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJndE00Y1dCVmlpRWVFTmIySzJrUU1wemFTYlVxNDVGRSJ9._ATcBvOhS5fZzkGeCrjAUlOWNFYil2KIAo7mlcPReUE",
    base_url="https://ai-dcin-test.digitalyili.com/v1"
)

print("创建agent...")
agent = create_deep_agent(model=model)

print("调用agent...")
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print("\n结果:", result)
