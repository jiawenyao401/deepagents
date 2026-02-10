from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

print("配置模型...")
model = ChatOpenAI(
    model="ms-7s6gfd9v",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJndE00Y1dCVmlpRWVFTmIySzJrUU1wemFTYlVxNDVGRSJ9._ATcBvOhS5fZzkGeCrjAUlOWNFYil2KIAo7mlcPReUE",
    base_url="https://ai-dcin-test.digitalyili.com/v1"
)

# 禁用多模态支持，强制使用字符串 content
model.bind(strict=True)

print("创建agent...")
agent = create_deep_agent(
    model=model
)

print("调用agent...")
result = agent.invoke({"messages": [{"role": "user", "content": "你好，请介绍一下自己"}]})
print("\n结果:", result)
