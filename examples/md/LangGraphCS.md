# LangGraph Client-Server 通信架构详解

## 目录
1. [架构概述](#架构概述)
2. [通信流程](#通信流程)
3. [核心组件](#核心组件)
4. [实现细节](#实现细节)
5. [代码示例](#代码示例)
6. [状态管理](#状态管理)
7. [错误处理](#错误处理)

---

## 架构概述

LangGraph采用**Client-Server分离架构**，支持分布式部署和远程执行。

### 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                      Client Layer                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LangGraph SDK Client                                │   │
│  │  - Thread Management                                 │   │
│  │  - Run Execution                                     │   │
│  │  - Event Streaming                                  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                           ↕
                    HTTP / WebSocket
                           ↕
┌─────────────────────────────────────────────────────────────┐
│                      Server Layer                           │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  LangGraph Server                                    │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │ Graph Runtime                                  │  │   │
│  │  │ - Node Execution                              │  │   │
│  │  │ - State Management                            │  │   │
│  │  │ - Tool Invocation                             │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  │  ┌────────────────────────────────────────────────┐  │   │
│  │  │ Storage Layer                                  │  │   │
│  │  │ - Thread State Persistence                    │  │   │
│  │  │ - Checkpoint Management                       │  │   │
│  │  │ - History Offloading                          │  │   │
│  │  └────────────────────────────────────────────────┘  │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 核心概念

| 概念 | 说明 |
|------|------|
| **Thread** | 对话线程，维护独立的执行上下文和状态 |
| **Run** | 单次执行实例，包含输入、输出和中间步骤 |
| **Assistant** | 编译后的LangGraph图，可被多个线程调用 |
| **Event** | 执行过程中产生的事件流（节点执行、工具调用等） |
| **Checkpoint** | 执行状态快照，支持恢复和重放 |

---

## 通信流程

### 完整请求-响应周期

```
1. 初始化阶段
   ├─ Client: 连接到Server (HTTP/WebSocket)
   └─ Server: 验证连接，加载图定义

2. 线程创建阶段
   ├─ Client: POST /threads
   ├─ Server: 创建新线程，分配thread_id
   └─ Client: 接收thread_id

3. 执行阶段
   ├─ Client: POST /runs/stream (发送输入)
   ├─ Server: 
   │  ├─ 解析输入消息
   │  ├─ 初始化运行上下文
   │  └─ 执行图的节点序列
   └─ Client: 通过WebSocket接收事件流

4. 事件流处理
   ├─ Server发送事件:
   │  ├─ on_node_start: 节点开始执行
   │  ├─ on_node_end: 节点执行完成
   │  ├─ on_tool_call: 工具调用
   │  ├─ on_tool_result: 工具结果
   │  └─ on_run_end: 运行完成
   └─ Client: 实时处理和显示事件

5. 状态持久化
   ├─ Server: 保存线程状态到存储
   ├─ Server: 保存检查点
   └─ Server: 可选的历史摘要化
```

### 时序图

```
Client                          Server                    Storage
  │                               │                          │
  ├──────── POST /threads ───────→│                          │
  │                               ├─ Create Thread ─────────→│
  │                               │                          │
  │←─ thread_id ─────────────────┤                          │
  │                               │                          │
  ├─ POST /runs/stream ──────────→│                          │
  │  (input messages)             │                          │
  │                               ├─ Initialize Run ────────→│
  │                               │                          │
  │                               ├─ Execute Node 1         │
  │                               ├─ Execute Node 2         │
  │                               ├─ Call Tool              │
  │                               │                          │
  │←─ Event: on_node_start ──────┤                          │
  │←─ Event: on_tool_call ───────┤                          │
  │←─ Event: on_tool_result ─────┤                          │
  │←─ Event: on_node_end ────────┤                          │
  │                               ├─ Save Checkpoint ──────→│
  │                               │                          │
  │←─ Event: on_run_end ─────────┤                          │
  │                               ├─ Persist State ────────→│
  │                               │                          │
```

---

## 核心组件

### 1. Client 组件

```python
from langgraph_sdk import get_client

class LangGraphClient:
    """LangGraph SDK客户端"""
    
    def __init__(self, url: str):
        """初始化客户端连接"""
        self.url = url
        self.session = None
    
    def connect(self):
        """建立连接"""
        # HTTP连接用于REST API
        # WebSocket连接用于事件流
        pass
    
    def threads(self):
        """线程管理接口"""
        # create() - 创建新线程
        # get() - 获取线程信息
        # list() - 列出所有线程
        pass
    
    def runs(self):
        """运行管理接口"""
        # stream() - 流式执行
        # create() - 创建运行
        # get() - 获取运行状态
        pass
```

### 2. Server 组件

```python
from langgraph_sdk import LangGraphServer

class LangGraphServer:
    """LangGraph服务器"""
    
    def __init__(self, graphs: dict, port: int = 2024):
        """初始化服务器
        
        Args:
            graphs: 编译后的图字典 {"assistant_id": compiled_graph}
            port: 监听端口
        """
        self.graphs = graphs
        self.port = port
        self.storage = None  # 状态存储
    
    def run(self):
        """启动服务器"""
        # 启动HTTP服务
        # 启动WebSocket服务
        # 加载图定义
        pass
```

### 3. 通信协议

#### REST API 端点

| 方法 | 端点 | 说明 |
|------|------|------|
| POST | `/threads` | 创建新线程 |
| GET | `/threads/{thread_id}` | 获取线程信息 |
| GET | `/threads` | 列出所有线程 |
| POST | `/runs/stream` | 流式执行（WebSocket升级） |
| GET | `/runs/{run_id}` | 获取运行状态 |
| POST | `/runs/{run_id}/cancel` | 取消运行 |

#### 消息格式

**创建线程请求：**
```json
{
  "metadata": {
    "user_id": "user123",
    "session_name": "research_session"
  }
}
```

**创建线程响应：**
```json
{
  "thread_id": "thread_abc123",
  "created_at": "2024-01-15T10:30:00Z",
  "metadata": {}
}
```

**执行请求：**
```json
{
  "assistant_id": "agent",
  "thread_id": "thread_abc123",
  "input": {
    "messages": [
      {
        "role": "user",
        "content": "Research LangGraph"
      }
    ]
  },
  "config": {
    "configurable": {
      "model": "gpt-4o"
    }
  }
}
```

**事件流格式：**
```json
{
  "event": "on_node_start",
  "data": {
    "node": "agent",
    "input": {...}
  }
}
```

---

## 实现细节

### 1. 线程管理

```python
class ThreadManager:
    """线程生命周期管理"""
    
    def create_thread(self, metadata: dict = None) -> str:
        """创建线程
        
        Returns:
            thread_id: 唯一线程标识
        """
        thread_id = generate_uuid()
        self.storage.save_thread({
            "thread_id": thread_id,
            "created_at": datetime.now(),
            "metadata": metadata or {},
            "state": {}
        })
        return thread_id
    
    def get_thread_state(self, thread_id: str) -> dict:
        """获取线程当前状态"""
        return self.storage.load_thread(thread_id)
    
    def update_thread_state(self, thread_id: str, state: dict):
        """更新线程状态"""
        self.storage.save_thread_state(thread_id, state)
```

### 2. 运行执行

```python
class RunExecutor:
    """运行执行引擎"""
    
    async def stream_run(
        self,
        thread_id: str,
        assistant_id: str,
        input_data: dict,
        config: dict = None
    ):
        """流式执行运行
        
        Yields:
            Event: 执行过程中的事件
        """
        # 1. 加载线程状态
        thread_state = self.thread_manager.get_thread_state(thread_id)
        
        # 2. 获取图定义
        graph = self.graphs[assistant_id]
        
        # 3. 创建运行上下文
        run_context = RunContext(
            thread_id=thread_id,
            run_id=generate_uuid(),
            graph=graph,
            state=thread_state
        )
        
        # 4. 执行图
        async for event in graph.astream(
            input_data,
            config=config,
            stream_mode="updates"
        ):
            # 5. 发送事件到客户端
            yield event
            
            # 6. 保存检查点
            self.checkpoint_manager.save(run_context)
        
        # 7. 保存最终状态
        self.thread_manager.update_thread_state(
            thread_id,
            run_context.state
        )
```

### 3. 事件流处理

```python
class EventStreamHandler:
    """事件流处理器"""
    
    async def handle_stream(self, websocket, run_generator):
        """处理事件流
        
        Args:
            websocket: WebSocket连接
            run_generator: 运行事件生成器
        """
        try:
            async for event in run_generator:
                # 转换事件格式
                formatted_event = self.format_event(event)
                
                # 发送到客户端
                await websocket.send_json(formatted_event)
                
                # 记录事件
                logger.debug(f"Event sent: {formatted_event['event']}")
                
        except Exception as e:
            # 发送错误事件
            await websocket.send_json({
                "event": "on_error",
                "data": {"error": str(e)}
            })
    
    def format_event(self, event: dict) -> dict:
        """格式化事件"""
        return {
            "event": event.get("event"),
            "data": event.get("data"),
            "timestamp": datetime.now().isoformat()
        }
```

### 4. 状态持久化

```python
class StateStorage:
    """状态存储层"""
    
    def save_thread(self, thread_data: dict):
        """保存线程数据"""
        # 支持多种存储后端：
        # - 内存存储（开发）
        # - 文件系统（本地）
        # - 数据库（生产）
        # - Redis（分布式）
        pass
    
    def load_thread(self, thread_id: str) -> dict:
        """加载线程数据"""
        pass
    
    def save_checkpoint(self, checkpoint: dict):
        """保存检查点"""
        pass
    
    def load_checkpoint(self, checkpoint_id: str) -> dict:
        """加载检查点"""
        pass
```

---

## 代码示例

### 示例1：基础Client使用

```python
from langgraph_sdk import get_client

# 连接到服务器
client = get_client(url="http://localhost:2024")

# 创建线程
thread = client.threads.create()
print(f"Thread ID: {thread['thread_id']}")

# 流式执行
for event in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id="agent",
    input={"messages": [{"role": "user", "content": "Research LangGraph"}]},
):
    print(f"Event: {event['event']}")
    if event['event'] == 'on_tool_call':
        print(f"Tool: {event['data']['tool']}")
    elif event['event'] == 'on_node_end':
        print(f"Node output: {event['data']['output']}")
```

### 示例2：Server启动

```python
from langgraph_sdk import LangGraphServer
from langgraph.graph import StateGraph
from langgraph.types import StateSnapshot
from typing_extensions import TypedDict

# 定义状态
class State(TypedDict):
    messages: list

# 构建图
def agent_node(state: State):
    # 处理逻辑
    return {"messages": state["messages"]}

graph = StateGraph(State).add_node("agent", agent_node).compile()

# 启动服务器
server = LangGraphServer(
    graphs={"agent": graph},
    port=2024
)

if __name__ == "__main__":
    server.run()
```

### 示例3：自定义配置

```python
from langgraph_sdk import get_client

client = get_client(url="http://localhost:2024")

# 创建线程并指定元数据
thread = client.threads.create(
    metadata={
        "user_id": "user123",
        "project": "research"
    }
)

# 执行时指定配置
for event in client.runs.stream(
    thread_id=thread["thread_id"],
    assistant_id="agent",
    input={"messages": [...]},
    config={
        "configurable": {
            "model": "gpt-4o",
            "temperature": 0.7
        }
    }
):
    print(event)
```

### 示例4：错误处理和重试

```python
import asyncio
from langgraph_sdk import get_client

async def execute_with_retry(client, thread_id, max_retries=3):
    """带重试的执行"""
    for attempt in range(max_retries):
        try:
            async for event in client.runs.stream(
                thread_id=thread_id,
                assistant_id="agent",
                input={"messages": [...]},
            ):
                yield event
            return
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避
                print(f"Retry after {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
            else:
                raise

# 使用
client = get_client(url="http://localhost:2024")
thread = client.threads.create()

async for event in execute_with_retry(client, thread["thread_id"]):
    print(event)
```

---

## 状态管理

### 状态生命周期

```
创建 → 初始化 → 执行中 → 检查点 → 完成 → 持久化
 │                                        │
 └────────────────────────────────────────┘
           可恢复状态
```

### 检查点机制

```python
class CheckpointManager:
    """检查点管理"""
    
    def save_checkpoint(self, run_id: str, state: dict, step: int):
        """保存检查点
        
        Args:
            run_id: 运行ID
            state: 当前状态
            step: 执行步骤
        """
        checkpoint = {
            "run_id": run_id,
            "step": step,
            "state": state,
            "timestamp": datetime.now(),
            "checkpoint_id": f"{run_id}_{step}"
        }
        self.storage.save(checkpoint)
    
    def load_checkpoint(self, checkpoint_id: str) -> dict:
        """加载检查点"""
        return self.storage.load(checkpoint_id)
    
    def resume_from_checkpoint(self, checkpoint_id: str):
        """从检查点恢复"""
        checkpoint = self.load_checkpoint(checkpoint_id)
        return checkpoint["state"]
```

### 状态同步

```python
class StateSynchronizer:
    """状态同步器"""
    
    async def sync_state(self, thread_id: str, state: dict):
        """同步状态到存储"""
        # 1. 序列化状态
        serialized = self.serialize(state)
        
        # 2. 保存到存储
        await self.storage.save(thread_id, serialized)
        
        # 3. 更新缓存
        self.cache.set(thread_id, state)
        
        # 4. 发送通知
        await self.notify_subscribers(thread_id, state)
```

---

## 错误处理

### 错误类型

| 错误类型 | HTTP状态码 | 说明 |
|---------|-----------|------|
| ValidationError | 400 | 输入验证失败 |
| ThreadNotFound | 404 | 线程不存在 |
| AssistantNotFound | 404 | 助手不存在 |
| ExecutionError | 500 | 执行过程中出错 |
| TimeoutError | 504 | 执行超时 |

### 错误处理策略

```python
class ErrorHandler:
    """错误处理器"""
    
    async def handle_error(self, error: Exception, context: dict):
        """处理错误
        
        Args:
            error: 异常对象
            context: 错误上下文
        """
        # 1. 记录错误
        logger.error(f"Error: {error}", extra=context)
        
        # 2. 确定错误类型
        error_type = self.classify_error(error)
        
        # 3. 生成错误响应
        response = self.generate_error_response(error_type, error)
        
        # 4. 清理资源
        await self.cleanup(context)
        
        # 5. 返回错误响应
        return response
    
    def classify_error(self, error: Exception) -> str:
        """分类错误"""
        if isinstance(error, ValueError):
            return "validation_error"
        elif isinstance(error, TimeoutError):
            return "timeout_error"
        else:
            return "execution_error"
```

---

## 最佳实践

### 1. 连接管理

```python
# ✅ 好的做法：使用上下文管理器
async with get_client(url="http://localhost:2024") as client:
    thread = client.threads.create()
    # 使用client
# 自动关闭连接

# ❌ 避免：手动管理连接
client = get_client(url="http://localhost:2024")
# 容易忘记关闭
```

### 2. 事件处理

```python
# ✅ 好的做法：处理所有事件类型
for event in client.runs.stream(...):
    if event['event'] == 'on_node_start':
        handle_node_start(event)
    elif event['event'] == 'on_tool_call':
        handle_tool_call(event)
    elif event['event'] == 'on_error':
        handle_error(event)

# ❌ 避免：忽略错误事件
for event in client.runs.stream(...):
    if event['event'] == 'on_node_end':
        process_output(event)
```

### 3. 状态管理

```python
# ✅ 好的做法：定期保存检查点
async def execute_long_task(client, thread_id):
    checkpoint_interval = 10  # 每10步保存
    step = 0
    
    for event in client.runs.stream(...):
        step += 1
        if step % checkpoint_interval == 0:
            save_checkpoint(thread_id, step)

# ❌ 避免：只在最后保存
for event in client.runs.stream(...):
    pass
# 如果中途失败，所有进度丢失
```

### 4. 超时处理

```python
# ✅ 好的做法：设置超时
import asyncio

try:
    async with asyncio.timeout(300):  # 5分钟超时
        async for event in client.runs.stream(...):
            process_event(event)
except asyncio.TimeoutError:
    print("Execution timeout")

# ❌ 避免：无限等待
async for event in client.runs.stream(...):
    process_event(event)
```

---

## 总结

LangGraph Client-Server架构提供了：

- **分离关注点**：Client处理UI/交互，Server处理执行逻辑
- **可扩展性**：支持多个Client连接到单个Server
- **容错性**：检查点和状态持久化支持恢复
- **实时性**：WebSocket事件流提供实时反馈
- **灵活性**：支持自定义存储、中间件和工具

通过理解这些通信机制，你可以构建健壮的分布式Agent系统。
