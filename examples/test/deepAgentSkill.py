from deepagents import create_deep_agent
from langchain_openai import ChatOpenAI
from deepagents.backends.filesystem import FilesystemBackend

fs_backend = FilesystemBackend(root_dir="D:\\test", virtual_mode=True)

model = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJ5RXpSUUpjVk1sODFNZVZjNjJPcG9VQXkwTHdRSWNOdiJ9.J-G-doiKv0dgYW9vymVU_PIfLtY1J2fynt_IEKhHrtU",
    base_url="https://ai-dcin-test.digitalyili.com/v1"
)

agent = create_deep_agent(
    model=model,
    backend=fs_backend,
    skills=["/skill/skills/skills/"]  # 注意末尾的斜杠
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": """帮我创建一个伊利QQ星奶的宣传PPT，保存到 /伊利QQ星奶的ppt.pptx
        
要求：
1. 使用 python-pptx 库创建（如果没有安装，先用 execute 工具安装：pip install python-pptx）
2. 创建一个简单的 Python 脚本来生成 PPTX
3. 包含 3-5 张幻灯片
4. 执行脚本生成文件
"""
    }]
})

# 打印最后的消息查看 Agent 的回复
print("\n=== Agent Response ===")
for msg in result["messages"][-5:]:
    if hasattr(msg, "content") and msg.content:
        content_str = str(msg.content)
        print(f"\n[{msg.__class__.__name__}]: {content_str[:500]}..." if len(content_str) > 500 else f"\n[{msg.__class__.__name__}]: {content_str}")

# 检查文件是否创建
import os
print("\n=== File Check ===")
if os.path.exists("D:\\test\\伊利QQ星奶的ppt.pptx"):
    print("[OK] PPTX file created successfully!")
else:
    print("[FAIL] PPTX file was NOT created")