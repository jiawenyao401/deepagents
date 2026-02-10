from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI

# ===== 示例 1: 通义千问 (阿里云) =====
model_qwen = ChatOpenAI(
    model="qwen-plus",
    api_key="sk-xxx",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# ===== 示例 2: DeepSeek =====
model_deepseek = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-xxx",
    base_url="https://api.deepseek.com/v1"
)

# ===== 示例 3: 智谱 AI (GLM) =====
model_glm = ChatOpenAI(
    model="glm-4",
    api_key="xxx.xxx",
    base_url="https://open.bigmodel.cn/api/paas/v4"
)

# ===== 示例 4: 月之暗面 (Moonshot/Kimi) =====
model_moonshot = ChatOpenAI(
    model="moonshot-v1-8k",
    api_key="sk-xxx",
    base_url="https://api.moonshot.cn/v1"
)

# ===== 示例 5: 本地 Ollama =====
model_ollama = ChatOpenAI(
    model="llama2",
    api_key="ollama",  # Ollama 不需要真实 key
    base_url="http://localhost:11434/v1"
)

# 使用任意一个模型创建 agent
agent = create_deep_agent(model=model_qwen)
result = agent.invoke({"messages": [{"role": "user", "content": "你好"}]})
print(result)
