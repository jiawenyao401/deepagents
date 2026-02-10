# -*- coding: utf-8 -*-
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
    backend=fs_backend
)

# 简单测试：写入一个文本文件
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "创建一个文件 /test.txt，内容是：Hello World"
    }]
})

print("\n=== Checking if file was created ===")
import os
if os.path.exists("D:\\test\\test.txt"):
    print("[OK] File created successfully!")
    with open("D:\\test\\test.txt", "r") as f:
        print(f"Content: {f.read()}")
else:
    print("[FAIL] File was NOT created")
    print("\n=== Last few messages ===")
    for msg in result["messages"][-3:]:
        if hasattr(msg, "content"):
            print(f"\n[{msg.__class__.__name__}]: {str(msg.content)[:300]}")
