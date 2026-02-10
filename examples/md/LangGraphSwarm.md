# LangGraphSwarm 详解

## 目录
1. [概念介绍](#概念介绍)
2. [架构设计](#架构设计)
3. [核心组件](#核心组件)
4. [工作流程](#工作流程)
5. [代码示例](#代码示例)
6. [最佳实践](#最佳实践)

---

## 概念介绍

### 什么是LangGraphSwarm？

LangGraphSwarm是一个基于LangGraph的**多Agent协作框架**，用于构建能够相互通信、协调和委派任务的Agent群体系统。

### 核心特性

| 特性 | 说明 |
|------|------|
| **多Agent协作** | 支持多个Agent并行或串行执行 |
| **任务委派** | Agent可以将任务委派给其他Agent |
| **上下文共享** | Agent之间可以共享执行上下文 |
| **动态路由** | 根据任务类型动态选择合适的Agent |
| **状态同步** | 维护全局状态和局部状态的一致性 |
| **错误恢复** | 支持失败重试和降级处理 |

### 应用场景

- **研究助手**：多个Agent分别负责搜索、分析、总结
- **内容创作**：规划Agent、写作Agent、编辑Agent协作
- **数据处理**：提取Agent、转换Agent、验证Agent流水线
- **客服系统**：分类Agent、处理Agent、升级Agent协作
- **代码审查**：静态分析Agent、测试Agent、文档Agent

---

## 架构设计

### 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Swarm Orchestrator                       │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Task Router & Dispatcher                             │   │
│  │ - 任务分类                                            │   │
│  │ - Agent选择                                          │   │
│  │ - 优先级管理                                          │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↕
        ┌──────────────────┼──────────────────┐
        ↓                  ↓                  ↓
   ┌─────────┐        ┌─────────┐        ┌─────────┐
   │ Agent 1 │        │ Agent 2 │        │ Agent N │
   │ (Search)│        │(Analyze)│        │(Summary)│
   └─────────┘        └─────────┘        └─────────┘
        ↓                  ↓                  ↓
   ┌─────────────────────────────────────────────────┐
   │         Shared Context & State Manager          │
   │ - 全局状态                                       │
   │ - 消息队列                                       │
   │ - 执行历史                                       │
   └─────────────────────────────────────────────────┘
        ↓
   ┌─────────────────────────────────────────────────┐
   │         Tool & Resource Layer                   │
   │ - 文件系统                                       │
   │ - 外部API                                       │
   │ - 数据库                                        │
   └─────────────────────────────────────────────────┘
```

### Agent生命周期

```
初始化 → 等待任务 → 接收任务 → 执行 → 结果处理 → 状态更新 → 完成
  │                                                      │
  └──────────────────────────────────────────────────────┘
                    可循环执行
```

---

## 核心组件

### 1. Swarm Orchestrator（群体编排器）

```python
from typing import Any, Callable, Optional
from dataclasses import dataclass
from enum import Enum

class TaskPriority(Enum):
    """任务优先级"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4

@dataclass
class Task:
    """任务定义"""
    id: str
    type: str
    content: str
    priority: TaskPriority = TaskPriority.NORMAL
    metadata: dict = None
    assigned_agent: Optional[str] = None

class SwarmOrchestrator:
    """群体编排器"""
    
    def __init__(self):
        self.agents: dict[str, "Agent"] = {}
        self.task_queue: list[Task] = []
        self.task_router: dict[str, str] = {}  # 任务类型 -> Agent ID
        self.context: dict[str, Any] = {}
    
    def register_agent(self, agent_id: str, agent: "Agent"):
        """注册Agent"""
        self.agents[agent_id] = agent
    
    def register_task_route(self, task_type: str, agent_id: str):
        """注册任务路由"""
        self.task_router[task_type] = agent_id
    
    def submit_task(self, task: Task):
        """提交任务"""
        # 根据优先级排序
        self.task_queue.append(task)
        self.task_queue.sort(key=lambda t: t.priority.value, reverse=True)
    
    async def dispatch_task(self, task: Task) -> Any:
        """分派任务"""
        # 1. 确定目标Agent
        agent_id = self.task_router.get(task.type)
        if not agent_id:
            raise ValueError(f"No agent for task type: {task.type}")
        
        # 2. 获取Agent
        agent = self.agents[agent_id]
        
        # 3. 执行任务
        result = await agent.execute(task, self.context)
        
        # 4. 更新上下文
        self.context.update(result.get("context_update", {}))
        
        return result
    
    async def run(self):
        """运行群体"""
        while self.task_queue:
            task = self.task_queue.pop(0)
            try:
                result = await self.dispatch_task(task)
                print(f"Task {task.id} completed: {result}")
            except Exception as e:
                print(f"Task {task.id} failed: {e}")
```

### 2. Agent（代理）

```python
from abc import ABC, abstractmethod
from typing import Any, Dict

class Agent(ABC):
    """Agent基类"""
    
    def __init__(self, agent_id: str, name: str, description: str):
        self.agent_id = agent_id
        self.name = name
        self.description = description
        self.tools: list[Callable] = []
        self.state: dict[str, Any] = {}
    
    def add_tool(self, tool: Callable):
        """添加工具"""
        self.tools.append(tool)
    
    @abstractmethod
    async def execute(self, task: Task, context: dict) -> dict:
        """执行任务"""
        pass
    
    async def delegate(self, subtask: Task, orchestrator: SwarmOrchestrator) -> Any:
        """委派子任务"""
        return await orchestrator.dispatch_task(subtask)

class SpecializedAgent(Agent):
    """专业化Agent"""
    
    def __init__(self, agent_id: str, name: str, description: str, 
                 system_prompt: str):
        super().__init__(agent_id, name, description)
        self.system_prompt = system_prompt
        self.model = None  # LLM模型
    
    async def execute(self, task: Task, context: dict) -> dict:
        """执行任务"""
        # 1. 准备输入
        input_data = {
            "task": task.content,
            "context": context,
            "tools": [t.__name__ for t in self.tools]
        }
        
        # 2. 调用LLM
        response = await self.model.agenerate(
            system_prompt=self.system_prompt,
            user_input=input_data
        )
        
        # 3. 处理响应
        result = self._parse_response(response)
        
        # 4. 执行工具调用
        for tool_call in result.get("tool_calls", []):
            tool_result = await self._execute_tool(tool_call)
            result["tool_results"].append(tool_result)
        
        return result
    
    async def _execute_tool(self, tool_call: dict) -> Any:
        """执行工具"""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        for tool in self.tools:
            if tool.__name__ == tool_name:
                return await tool(**tool_args)
        
        raise ValueError(f"Tool not found: {tool_name}")
    
    def _parse_response(self, response: str) -> dict:
        """解析响应"""
        # 解析LLM输出，提取工具调用和最终答案
        pass
```

### 3. Context Manager（上下文管理器）

```python
from datetime import datetime
from typing import Any

class ContextManager:
    """上下文管理器"""
    
    def __init__(self):
        self.global_context: dict[str, Any] = {}
        self.agent_contexts: dict[str, dict[str, Any]] = {}
        self.message_history: list[dict] = []
        self.execution_trace: list[dict] = []
    
    def set_global(self, key: str, value: Any):
        """设置全局上下文"""
        self.global_context[key] = value
    
    def get_global(self, key: str, default: Any = None) -> Any:
        """获取全局上下文"""
        return self.global_context.get(key, default)
    
    def set_agent_context(self, agent_id: str, key: str, value: Any):
        """设置Agent上下文"""
        if agent_id not in self.agent_contexts:
            self.agent_contexts[agent_id] = {}
        self.agent_contexts[agent_id][key] = value
    
    def get_agent_context(self, agent_id: str, key: str, default: Any = None) -> Any:
        """获取Agent上下文"""
        if agent_id not in self.agent_contexts:
            return default
        return self.agent_contexts[agent_id].get(key, default)
    
    def add_message(self, agent_id: str, role: str, content: str):
        """添加消息"""
        self.message_history.append({
            "timestamp": datetime.now().isoformat(),
            "agent_id": agent_id,
            "role": role,
            "content": content
        })
    
    def add_trace(self, event: str, data: dict):
        """添加执行跟踪"""
        self.execution_trace.append({
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "data": data
        })
    
    def get_summary(self) -> dict:
        """获取上下文摘要"""
        return {
            "global_context": self.global_context,
            "agent_count": len(self.agent_contexts),
            "message_count": len(self.message_history),
            "trace_events": len(self.execution_trace)
        }
```

### 4. Task Router（任务路由器）

```python
from typing import Callable, Optional

class TaskRouter:
    """任务路由器"""
    
    def __init__(self):
        self.routes: dict[str, str] = {}  # 任务类型 -> Agent ID
        self.rules: list[tuple[Callable, str]] = []  # (条件, Agent ID)
        self.default_agent: Optional[str] = None
    
    def register_route(self, task_type: str, agent_id: str):
        """注册直接路由"""
        self.routes[task_type] = agent_id
    
    def register_rule(self, condition: Callable[[Task], bool], agent_id: str):
        """注册条件路由"""
        self.rules.append((condition, agent_id))
    
    def set_default_agent(self, agent_id: str):
        """设置默认Agent"""
        self.default_agent = agent_id
    
    def route(self, task: Task) -> str:
        """路由任务"""
        # 1. 检查直接路由
        if task.type in self.routes:
            return self.routes[task.type]
        
        # 2. 检查条件路由
        for condition, agent_id in self.rules:
            if condition(task):
                return agent_id
        
        # 3. 使用默认Agent
        if self.default_agent:
            return self.default_agent
        
        raise ValueError(f"No route for task: {task.type}")
```

---

## 工作流程

### 完整执行流程

```
1. 初始化阶段
   ├─ 创建Orchestrator
   ├─ 注册Agents
   ├─ 配置任务路由
   └─ 初始化上下文

2. 任务提交阶段
   ├─ 创建Task对象
   ├─ 设置优先级
   ├─ 提交到队列
   └─ 按优先级排序

3. 任务分派阶段
   ├─ 从队列取出任务
   ├─ 路由到目标Agent
   ├─ 传递上下文
   └─ 启动执行

4. Agent执行阶段
   ├─ 接收任务
   ├─ 调用LLM
   ├─ 执行工具
   ├─ 可能委派子任务
   └─ 返回结果

5. 结果处理阶段
   ├─ 收集结果
   ├─ 更新上下文
   ├─ 记录执行跟踪
   └─ 触发后续任务

6. 完成阶段
   ├─ 汇总所有结果
   ├─ 生成最终输出
   ├─ 保存执行历史
   └─ 清理资源
```

### 时序图

```
User                Orchestrator         Agent1          Agent2
  │                      │                 │               │
  ├─ Submit Task ───────→│                 │               │
  │                      ├─ Route ────────→│               │
  │                      │                 ├─ Execute      │
  │                      │                 ├─ Delegate ───→│
  │                      │                 │               ├─ Execute
  │                      │                 │               ├─ Return
  │                      │                 │←──────────────┤
  │                      │                 ├─ Continue     │
  │                      │                 ├─ Return ─────→│
  │                      │←────────────────┤               │
  │                      ├─ Update Context │               │
  │←─ Result ───────────┤                 │               │
```

---

## 代码示例

### 示例1：基础多Agent系统

```python
import asyncio
from typing import Any

# 定义Agent
class SearchAgent(SpecializedAgent):
    async def execute(self, task: Task, context: dict) -> dict:
        print(f"SearchAgent executing: {task.content}")
        # 模拟搜索
        results = [
            "Result 1: LangGraph is a framework...",
            "Result 2: Multi-agent systems..."
        ]
        return {
            "results": results,
            "context_update": {"search_results": results}
        }

class AnalysisAgent(SpecializedAgent):
    async def execute(self, task: Task, context: dict) -> dict:
        print(f"AnalysisAgent executing: {task.content}")
        search_results = context.get("search_results", [])
        # 分析结果
        analysis = {
            "key_points": ["Point 1", "Point 2"],
            "summary": "Analysis summary"
        }
        return {
            "analysis": analysis,
            "context_update": {"analysis": analysis}
        }

class SummaryAgent(SpecializedAgent):
    async def execute(self, task: Task, context: dict) -> dict:
        print(f"SummaryAgent executing: {task.content}")
        analysis = context.get("analysis", {})
        # 生成摘要
        summary = "Final summary based on analysis"
        return {
            "summary": summary,
            "context_update": {"final_summary": summary}
        }

# 创建系统
async def main():
    # 初始化
    orchestrator = SwarmOrchestrator()
    
    # 创建Agents
    search_agent = SearchAgent("search", "Search Agent", "Searches for information")
    analysis_agent = AnalysisAgent("analysis", "Analysis Agent", "Analyzes information")
    summary_agent = SummaryAgent("summary", "Summary Agent", "Summarizes findings")
    
    # 注册Agents
    orchestrator.register_agent("search", search_agent)
    orchestrator.register_agent("analysis", analysis_agent)
    orchestrator.register_agent("summary", summary_agent)
    
    # 配置路由
    orchestrator.register_task_route("search", "search")
    orchestrator.register_task_route("analysis", "analysis")
    orchestrator.register_task_route("summary", "summary")
    
    # 创建任务
    tasks = [
        Task("1", "search", "Research LangGraph", TaskPriority.HIGH),
        Task("2", "analysis", "Analyze search results", TaskPriority.NORMAL),
        Task("3", "summary", "Summarize findings", TaskPriority.NORMAL)
    ]
    
    # 提交任务
    for task in tasks:
        orchestrator.submit_task(task)
    
    # 执行
    while orchestrator.task_queue:
        task = orchestrator.task_queue.pop(0)
        result = await orchestrator.dispatch_task(task)
        print(f"Result: {result}")

asyncio.run(main())
```

### 示例2：任务委派

```python
class CoordinatorAgent(SpecializedAgent):
    """协调Agent - 可以委派子任务"""
    
    async def execute(self, task: Task, context: dict) -> dict:
        print(f"CoordinatorAgent: {task.content}")
        
        # 获取Orchestrator（通过context传递）
        orchestrator = context.get("orchestrator")
        
        # 委派子任务
        subtask1 = Task("sub1", "search", "Find information")
        subtask2 = Task("sub2", "analysis", "Analyze information")
        
        result1 = await self.delegate(subtask1, orchestrator)
        result2 = await self.delegate(subtask2, orchestrator)
        
        # 合并结果
        combined = {
            "search_result": result1,
            "analysis_result": result2,
            "context_update": {
                "combined_results": {
                    "search": result1,
                    "analysis": result2
                }
            }
        }
        
        return combined
```

### 示例3：条件路由

```python
def setup_conditional_routing(orchestrator: SwarmOrchestrator):
    """设置条件路由"""
    
    # 根据优先级路由
    def high_priority_rule(task: Task) -> bool:
        return task.priority == TaskPriority.URGENT
    
    orchestrator.task_router.register_rule(
        high_priority_rule,
        "priority_agent"
    )
    
    # 根据内容路由
    def research_rule(task: Task) -> bool:
        return "research" in task.content.lower()
    
    orchestrator.task_router.register_rule(
        research_rule,
        "research_agent"
    )
    
    # 设置默认Agent
    orchestrator.task_router.set_default_agent("general_agent")
```

### 示例4：上下文共享

```python
async def context_sharing_example():
    """演示上下文共享"""
    
    context_manager = ContextManager()
    
    # 设置全局上下文
    context_manager.set_global("project_name", "LangGraph Research")
    context_manager.set_global("deadline", "2024-12-31")
    
    # Agent设置自己的上下文
    context_manager.set_agent_context("search", "query_count", 0)
    context_manager.set_agent_context("analysis", "analysis_depth", "deep")
    
    # 添加消息
    context_manager.add_message("search", "system", "Starting search")
    context_manager.add_message("search", "result", "Found 10 results")
    
    # 添加执行跟踪
    context_manager.add_trace("task_start", {"task_id": "1", "agent": "search"})
    context_manager.add_trace("task_end", {"task_id": "1", "status": "success"})
    
    # 获取摘要
    summary = context_manager.get_summary()
    print(f"Context Summary: {summary}")
```

---

## 最佳实践

### 1. Agent设计

```python
# ✅ 好的做法：单一职责
class EmailAgent(SpecializedAgent):
    """专门处理邮件相关任务"""
    async def execute(self, task: Task, context: dict) -> dict:
        if task.type == "send_email":
            return await self._send_email(task)
        elif task.type == "parse_email":
            return await self._parse_email(task)

# ❌ 避免：职责过多
class UniversalAgent(SpecializedAgent):
    """处理所有任务"""
    async def execute(self, task: Task, context: dict) -> dict:
        # 处理邮件、文件、数据库等所有操作
        pass
```

### 2. 任务设计

```python
# ✅ 好的做法：清晰的任务定义
task = Task(
    id="research_001",
    type="research",
    content="Research LangGraph architecture",
    priority=TaskPriority.HIGH,
    metadata={
        "keywords": ["LangGraph", "architecture"],
        "max_results": 10,
        "language": "en"
    }
)

# ❌ 避免：模糊的任务定义
task = Task(
    id="task1",
    type="do_something",
    content="Do research"
)
```

### 3. 错误处理

```python
# ✅ 好的做法：完善的错误处理
async def execute_with_retry(orchestrator, task, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await orchestrator.dispatch_task(task)
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Retry after {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                print(f"Task failed after {max_retries} attempts")
                raise

# ❌ 避免：忽略错误
result = await orchestrator.dispatch_task(task)
```

### 4. 性能优化

```python
# ✅ 好的做法：并行执行独立任务
async def parallel_execution(orchestrator, tasks):
    # 过滤出独立任务
    independent_tasks = [t for t in tasks if not t.depends_on]
    
    # 并行执行
    results = await asyncio.gather(
        *[orchestrator.dispatch_task(t) for t in independent_tasks]
    )
    
    return results

# ❌ 避免：串行执行所有任务
for task in tasks:
    result = await orchestrator.dispatch_task(task)
```

### 5. 监控和日志

```python
# ✅ 好的做法：详细的监控
class MonitoredOrchestrator(SwarmOrchestrator):
    async def dispatch_task(self, task: Task) -> Any:
        start_time = time.time()
        
        try:
            result = await super().dispatch_task(task)
            duration = time.time() - start_time
            
            logger.info(
                f"Task {task.id} completed",
                extra={
                    "task_type": task.type,
                    "duration": duration,
                    "status": "success"
                }
            )
            
            return result
        except Exception as e:
            duration = time.time() - start_time
            
            logger.error(
                f"Task {task.id} failed",
                extra={
                    "task_type": task.type,
                    "duration": duration,
                    "error": str(e)
                }
            )
            
            raise
```

---

## 总结

LangGraphSwarm提供了一个强大的框架来构建多Agent系统：

- **灵活的架构**：支持多种Agent协作模式
- **任务管理**：优先级队列和智能路由
- **上下文共享**：全局和局部上下文管理
- **可扩展性**：易于添加新Agent和工具
- **可观测性**：完整的执行跟踪和监控

通过合理使用这些组件，可以构建高效、可靠的分布式Agent系统。
