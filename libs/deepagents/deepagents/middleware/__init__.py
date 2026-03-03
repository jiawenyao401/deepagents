"""中间件模块，导出所有可用的中间件类。"""

# 文件系统中间件：提供 ls/read_file/write_file/edit_file/glob/grep/execute 等工具
from deepagents.middleware.filesystem import FilesystemMiddleware

# 记忆中间件：从 AGENTS.md 文件加载持久化上下文并注入系统提示
from deepagents.middleware.memory import MemoryMiddleware

# OpenAI 兼容中间件：将消息内容从数组格式转换为字符串格式，兼容旧版 OpenAI API
from deepagents.middleware.openai_compat import OpenAICompatMiddleware

# 技能中间件：加载并向系统提示暴露 Agent 技能（渐进式披露模式）
from deepagents.middleware.skills import SkillsMiddleware

# 子代理中间件：通过 task 工具向 Agent 提供子代理能力
from deepagents.middleware.subagents import CompiledSubAgent, SubAgent, SubAgentMiddleware

# 摘要中间件：自动压缩对话历史；工具版摘要中间件：提供手动触发压缩的工具
from deepagents.middleware.summarization import SummarizationMiddleware, SummarizationToolMiddleware

__all__ = [
    "CompiledSubAgent",
    "FilesystemMiddleware",
    "MemoryMiddleware",
    "OpenAICompatMiddleware",
    "SkillsMiddleware",
    "SubAgent",
    "SubAgentMiddleware",
    "SummarizationMiddleware",
    "SummarizationToolMiddleware",
]
