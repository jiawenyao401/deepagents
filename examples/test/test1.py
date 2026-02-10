import os
from deepagents import create_deep_agent

# 设置 API 密钥（请替换为你的实际密钥）
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"

agent = create_deep_agent()
result = agent.invoke({"messages": [{"role": "user", "content": "Research LangGraph and write a summary"}]})