from langchain_openai import ChatOpenAI

# 测试第三方API连接
model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJndE00Y1dCVmlpRWVFTmIySzJrUU1wemFTYlVxNDVGRSJ9._ATcBvOhS5fZzkGeCrjAUlOWNFYil2KIAo7mlcPReUE",
    base_url="https://ai-dcin-test.digitalyili.com/v1"
)

print("开始测试API...")
try:
    response = model.invoke("杭州的天气怎么样？")
    print("成功:", response.content)
except Exception as e:
    print("错误:", e)
