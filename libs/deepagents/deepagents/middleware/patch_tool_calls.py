"""修补消息历史中悬空工具调用的中间件。

当 AI 消息包含工具调用，但对应的 ToolMessage 响应缺失时（例如用户中断了工具调用），
本中间件会在 Agent 执行前自动补全这些"悬空"的工具调用，
防止因消息历史不完整而导致 API 报错。
"""

from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Overwrite


class PatchToolCallsMiddleware(AgentMiddleware):
    """修补消息历史中悬空工具调用的中间件。

    在 Agent 运行前检查所有 AIMessage，若某个工具调用没有对应的 ToolMessage 响应，
    则自动插入一条说明该工具调用已被取消的 ToolMessage，保证消息历史的完整性。
    """

    def before_agent(self, state: AgentState, runtime: Runtime[Any]) -> dict[str, Any] | None:  # noqa: ARG002
        """在 Agent 运行前处理所有 AIMessage 中的悬空工具调用。

        遍历消息历史，对每条 AIMessage 中的每个工具调用，
        检查后续消息中是否存在对应的 ToolMessage。
        若不存在，则插入一条"工具调用已取消"的 ToolMessage。

        Args:
            state: 当前 Agent 状态，包含消息历史。
            runtime: 运行时上下文（本方法未使用）。

        Returns:
            包含修补后消息列表的状态更新字典；若无需修补则返回 None。
        """
        messages = state["messages"]
        # 若消息列表为空，无需处理
        if not messages or len(messages) == 0:
            return None

        patched_messages = []
        # 遍历所有消息，检测并修补悬空工具调用
        for i, msg in enumerate(messages):
            patched_messages.append(msg)
            # 仅处理包含工具调用的 AI 消息
            if msg.type == "ai" and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    # 在当前消息之后的消息中查找对应的 ToolMessage
                    corresponding_tool_msg = next(
                        (msg for msg in messages[i:] if msg.type == "tool" and msg.tool_call_id == tool_call["id"]),  # ty: ignore[unresolved-attribute]
                        None,
                    )
                    if corresponding_tool_msg is None:
                        # 未找到对应响应，说明该工具调用是悬空的，插入取消说明
                        tool_msg = (
                            f"Tool call {tool_call['name']} with id {tool_call['id']} was "
                            "cancelled - another message came in before it could be completed."
                        )
                        patched_messages.append(
                            ToolMessage(
                                content=tool_msg,
                                name=tool_call["name"],
                                tool_call_id=tool_call["id"],
                            )
                        )

        # 使用 Overwrite 覆盖原消息列表，确保修补后的消息完整替换原列表
        return {"messages": Overwrite(patched_messages)}
