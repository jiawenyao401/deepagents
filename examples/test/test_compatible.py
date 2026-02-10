from deepagents import create_deep_agent
from deepagents.chat_models.openai_compat import CompatibleChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

# 配置模型 - 使用兼容旧版 OpenAI API 的包装器
# model = CompatibleChatOpenAI(
#     model="deepseek-ai/DeepSeek-V3",
#     appKey="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ5RXpSUUpjVk1sODFNZVZjNjJPcG9VQXkwTHdRSWNOdiJ9.J-G-doiKv0dgYW9vymVU_PIfLtY1J2fynt_IEKhHrtU",
#     base_url="https://ai-dcin-test.digitalyili.com/chatylserver",
#     temperature=0.7,
# )
model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ5RXpSUUpjVk1sODFNZVZjNjJPcG9VQXkwTHdRSWNOdiJ9.J-G-doiKv0dgYW9vymVU_PIfLtY1J2fynt_IEKhHrtU",
    base_url="https://ai-dcin-test.digitalyili.com/v1"

)


# 创建 agent
agent = create_deep_agent(
    model=model
)

result = agent.invoke({"messages": [{"role": "user", "content": "你好，请介绍一下自己"}]} )
print(result)
result = agent.invoke(
    {"messages": [{"role": "user", "content": "帮我写一个文件增删改查的python脚本"}]}
)
print(result)