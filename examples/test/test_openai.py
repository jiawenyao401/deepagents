from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJndE00Y1dCVmlpRWVFTmIySzJrUU1wemFTYlVxNDVGRSJ9._ATcBvOhS5fZzkGeCrjAUlOWNFYil2KIAo7mlcPReUE",
    base_url="https://ai-dcin-test.digitalyili.com/v1",
    model_kwargs={"stream": False}  # 禁用流式输出
)

agent = create_deep_agent(model=model)
result = agent.invoke({"messages": [{"role": "user", "content": "杭州的天气怎么样？"}]})
print(result)
