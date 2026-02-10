# LangGraph Agent 注册与发现机制详解

## 目录
1. [核心概念](#核心概念)
2. [注册机制](#注册机制)
3. [发现机制](#发现机制)
4. [实现方案](#实现方案)
5. [代码示例](#代码示例)
6. [最佳实践](#最佳实践)

---

## 核心概念

### 什么是Agent注册？

Agent注册是将Agent实例及其元数据存储到中央注册表中的过程，使系统能够：
- 追踪所有可用的Agent
- 快速查找和访问Agent
- 管理Agent的生命周期
- 支持动态Agent加载

### 什么是Agent发现？

Agent发现是根据特定条件（类型、能力、标签等）查找和定位Agent的过程。

### 核心流程

```
Agent创建 → 注册到Registry → 发布元数据 → 可被发现 → 被调用
   ↓                                              ↓
 初始化                                      执行任务
```

---

## 注册机制

### 1. 基础注册表

```python
from typing import Any, Dict, List, Optional, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import inspect

@dataclass
class AgentMetadata:
    """Agent元数据"""
    agent_id: str
    name: str
    description: str
    version: str = "1.0.0"
    tags: List[str] = field(default_factory=list)
    capabilities: List[str] = field(default_factory=list)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

class AgentRegistry:
    """Agent注册表"""
    
    def __init__(self):
        self._agents: Dict[str, Any] = {}
        self._metadata: Dict[str, AgentMetadata] = {}
        self._index: Dict[str, List[str]] = {}  # 标签索引
    
    def register(self, agent_id: str, agent: Any, metadata: AgentMetadata):
        """注册Agent"""
        if agent_id in self._agents:
            raise ValueError(f"Agent {agent_id} already registered")
        
        self._agents[agent_id] = agent
        self._metadata[agent_id] = metadata
        
        # 构建索引
        for tag in metadata.tags:
            if tag not in self._index:
                self._index[tag] = []
            self._index[tag].append(agent_id)
    
    def unregister(self, agent_id: str):
        """注销Agent"""
        if agent_id not in self._agents:
            raise ValueError(f"Agent {agent_id} not found")
        
        metadata = self._metadata[agent_id]
        
        # 清理索引
        for tag in metadata.tags:
            if tag in self._index:
                self._index[tag].remove(agent_id)
        
        del self._agents[agent_id]
        del self._metadata[agent_id]
    
    def get(self, agent_id: str) -> Optional[Any]:
        """获取Agent实例"""
        return self._agents.get(agent_id)
    
    def get_metadata(self, agent_id: str) -> Optional[AgentMetadata]:
        """获取Agent元数据"""
        return self._metadata.get(agent_id)
    
    def list_agents(self) -> List[str]:
        """列出所有Agent ID"""
        return list(self._agents.keys())
    
    def list_metadata(self) -> Dict[str, AgentMetadata]:
        """列出所有Agent元数据"""
        return dict(self._metadata)
```

### 2. 装饰器注册

```python
from functools import wraps
from datetime import datetime

class AgentRegistrar:
    """Agent注册器"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
    
    def register_agent(
        self,
        agent_id: str,
        name: str,
        description: str,
        tags: List[str] = None,
        capabilities: List[str] = None,
        version: str = "1.0.0"
    ):
        """装饰器：注册Agent"""
        def decorator(agent_class: Type) -> Type:
            # 提取schema
            input_schema = self._extract_schema(agent_class, "input")
            output_schema = self._extract_schema(agent_class, "output")
            
            # 创建元数据
            metadata = AgentMetadata(
                agent_id=agent_id,
                name=name,
                description=description,
                version=version,
                tags=tags or [],
                capabilities=capabilities or [],
                input_schema=input_schema,
                output_schema=output_schema,
                created_at=datetime.now().isoformat()
            )
            
            # 创建实例并注册
            agent_instance = agent_class()
            self.registry.register(agent_id, agent_instance, metadata)
            
            return agent_class
        
        return decorator
    
    def _extract_schema(self, agent_class: Type, schema_type: str) -> Dict:
        """提取schema"""
        method_name = f"get_{schema_type}_schema"
        if hasattr(agent_class, method_name):
            return getattr(agent_class, method_name)()
        return {}

# 使用示例
registry = AgentRegistry()
registrar = AgentRegistrar(registry)

@registrar.register_agent(
    agent_id="search_agent",
    name="Search Agent",
    description="Searches for information",
    tags=["search", "research"],
    capabilities=["web_search", "document_search"]
)
class SearchAgent:
    @staticmethod
    def get_input_schema():
        return {"query": "string", "max_results": "integer"}
    
    @staticmethod
    def get_output_schema():
        return {"results": "array", "total": "integer"}
```

### 3. 自动发现注册

```python
import importlib
import pkgutil
from pathlib import Path

class AutoDiscoveryRegistry(AgentRegistry):
    """自动发现注册表"""
    
    def auto_discover(self, package_path: str):
        """自动发现并注册Agent"""
        package = importlib.import_module(package_path)
        
        for importer, modname, ispkg in pkgutil.iter_modules(
            package.__path__,
            package.__name__ + "."
        ):
            module = importlib.import_module(modname)
            
            # 查找Agent类
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and hasattr(obj, "_agent_metadata"):
                    metadata = obj._agent_metadata
                    agent_instance = obj()
                    self.register(metadata.agent_id, agent_instance, metadata)
    
    def register_from_file(self, file_path: str):
        """从文件注册Agent"""
        spec = importlib.util.spec_from_file_location("agent_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and hasattr(obj, "_agent_metadata"):
                metadata = obj._agent_metadata
                agent_instance = obj()
                self.register(metadata.agent_id, agent_instance, metadata)
```

---

## 发现机制

### 1. 基础查询

```python
class AgentDiscovery:
    """Agent发现器"""
    
    def __init__(self, registry: AgentRegistry):
        self.registry = registry
    
    def find_by_id(self, agent_id: str) -> Optional[Any]:
        """按ID查找"""
        return self.registry.get(agent_id)
    
    def find_by_tag(self, tag: str) -> List[str]:
        """按标签查找"""
        return self.registry._index.get(tag, [])
    
    def find_by_capability(self, capability: str) -> List[str]:
        """按能力查找"""
        result = []
        for agent_id, metadata in self.registry._metadata.items():
            if capability in metadata.capabilities:
                result.append(agent_id)
        return result
    
    def find_by_name(self, name_pattern: str) -> List[str]:
        """按名称查找（支持模糊匹配）"""
        result = []
        for agent_id, metadata in self.registry._metadata.items():
            if name_pattern.lower() in metadata.name.lower():
                result.append(agent_id)
        return result
```

### 2. 高级查询

```python
from typing import Callable

class AdvancedDiscovery(AgentDiscovery):
    """高级发现器"""
    
    def find_by_filter(self, filter_func: Callable[[AgentMetadata], bool]) -> List[str]:
        """按自定义过滤器查找"""
        result = []
        for agent_id, metadata in self.registry._metadata.items():
            if filter_func(metadata):
                result.append(agent_id)
        return result
    
    def find_by_schema(self, input_schema: Dict) -> List[str]:
        """按输入schema查找兼容的Agent"""
        result = []
        for agent_id, metadata in self.registry._metadata.items():
            if self._schema_compatible(metadata.input_schema, input_schema):
                result.append(agent_id)
        return result
    
    def find_by_dependency(self, dependency: str) -> List[str]:
        """查找依赖特定资源的Agent"""
        result = []
        for agent_id, metadata in self.registry._metadata.items():
            if dependency in metadata.dependencies:
                result.append(agent_id)
        return result
    
    def _schema_compatible(self, agent_schema: Dict, input_schema: Dict) -> bool:
        """检查schema兼容性"""
        for key in input_schema:
            if key not in agent_schema:
                return False
        return True
```

### 3. 缓存和索引

```python
from functools import lru_cache
import hashlib

class CachedDiscovery(AdvancedDiscovery):
    """带缓存的发现器"""
    
    def __init__(self, registry: AgentRegistry):
        super().__init__(registry)
        self._cache: Dict[str, List[str]] = {}
        self._cache_version = 0
    
    def find_by_tag(self, tag: str) -> List[str]:
        """按标签查找（带缓存）"""
        cache_key = f"tag:{tag}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = super().find_by_tag(tag)
        self._cache[cache_key] = result
        return result
    
    def find_by_capability(self, capability: str) -> List[str]:
        """按能力查找（带缓存）"""
        cache_key = f"capability:{capability}"
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        result = super().find_by_capability(capability)
        self._cache[cache_key] = result
        return result
    
    def invalidate_cache(self):
        """清空缓存"""
        self._cache.clear()
        self._cache_version += 1
    
    def register_agent(self, agent_id: str, agent: Any, metadata: AgentMetadata):
        """注册时清空缓存"""
        super().register(agent_id, agent, metadata)
        self.invalidate_cache()
```

---

## 实现方案

### 方案1：中央注册表

```python
class CentralRegistry:
    """中央注册表（单例）"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.registry = AgentRegistry()
        self.discovery = CachedDiscovery(self.registry)
        self.registrar = AgentRegistrar(self.registry)
        self._initialized = True
    
    @classmethod
    def get_instance(cls):
        """获取单例"""
        return cls()

# 使用
central = CentralRegistry.get_instance()
agent = central.discovery.find_by_id("search_agent")
```

### 方案2：分布式注册表

```python
import asyncio
from typing import Set

class DistributedRegistry:
    """分布式注册表"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.local_registry = AgentRegistry()
        self.peer_registries: Dict[str, "DistributedRegistry"] = {}
        self.discovery = CachedDiscovery(self.local_registry)
    
    def register_peer(self, peer_id: str, peer_registry: "DistributedRegistry"):
        """注册对等节点"""
        self.peer_registries[peer_id] = peer_registry
    
    async def find_agent(self, agent_id: str) -> Optional[Any]:
        """查找Agent（本地优先）"""
        # 1. 本地查找
        agent = self.local_registry.get(agent_id)
        if agent:
            return agent
        
        # 2. 远程查找
        tasks = [
            peer.local_registry.get(agent_id)
            for peer in self.peer_registries.values()
        ]
        
        results = await asyncio.gather(*[asyncio.sleep(0) for _ in tasks])
        
        for result in results:
            if result:
                return result
        
        return None
    
    async def find_by_capability(self, capability: str) -> List[str]:
        """查找具有特定能力的Agent"""
        # 本地查找
        local_agents = self.discovery.find_by_capability(capability)
        
        # 远程查找
        remote_agents = []
        for peer in self.peer_registries.values():
            agents = peer.discovery.find_by_capability(capability)
            remote_agents.extend(agents)
        
        return local_agents + remote_agents
```

### 方案3：动态注册表

```python
from enum import Enum
import threading

class AgentState(Enum):
    """Agent状态"""
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    DEPRECATED = "deprecated"

class DynamicRegistry(AgentRegistry):
    """动态注册表"""
    
    def __init__(self):
        super().__init__()
        self._agent_states: Dict[str, AgentState] = {}
        self._lock = threading.RLock()
    
    def register(self, agent_id: str, agent: Any, metadata: AgentMetadata):
        """注册Agent"""
        with self._lock:
            super().register(agent_id, agent, metadata)
            self._agent_states[agent_id] = AgentState.REGISTERED
    
    def activate(self, agent_id: str):
        """激活Agent"""
        with self._lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent {agent_id} not found")
            self._agent_states[agent_id] = AgentState.ACTIVE
    
    def deactivate(self, agent_id: str):
        """停用Agent"""
        with self._lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent {agent_id} not found")
            self._agent_states[agent_id] = AgentState.INACTIVE
    
    def deprecate(self, agent_id: str):
        """标记Agent为已弃用"""
        with self._lock:
            if agent_id not in self._agents:
                raise ValueError(f"Agent {agent_id} not found")
            self._agent_states[agent_id] = AgentState.DEPRECATED
    
    def get_active_agents(self) -> List[str]:
        """获取所有活跃Agent"""
        with self._lock:
            return [
                agent_id for agent_id, state in self._agent_states.items()
                if state == AgentState.ACTIVE
            ]
```

---

## 代码示例

### 示例1：基础注册与发现

```python
# 创建注册表
registry = AgentRegistry()
discovery = AgentDiscovery(registry)

# 创建Agent
class SearchAgent:
    def search(self, query: str):
        return f"Searching for: {query}"

# 注册Agent
metadata = AgentMetadata(
    agent_id="search_001",
    name="Search Agent",
    description="Searches for information",
    tags=["search", "research"],
    capabilities=["web_search", "document_search"]
)

agent = SearchAgent()
registry.register("search_001", agent, metadata)

# 发现Agent
found_agent = discovery.find_by_id("search_001")
print(f"Found: {found_agent}")

# 按标签发现
search_agents = discovery.find_by_tag("search")
print(f"Search agents: {search_agents}")

# 按能力发现
web_search_agents = discovery.find_by_capability("web_search")
print(f"Web search agents: {web_search_agents}")
```

### 示例2：装饰器注册

```python
registry = AgentRegistry()
registrar = AgentRegistrar(registry)

@registrar.register_agent(
    agent_id="analysis_001",
    name="Analysis Agent",
    description="Analyzes data",
    tags=["analysis", "data"],
    capabilities=["statistical_analysis", "trend_analysis"]
)
class AnalysisAgent:
    @staticmethod
    def get_input_schema():
        return {"data": "array", "method": "string"}
    
    @staticmethod
    def get_output_schema():
        return {"result": "object", "confidence": "number"}
    
    def analyze(self, data, method):
        return {"result": "analysis result", "confidence": 0.95}

# 发现并使用
discovery = AgentDiscovery(registry)
agent_id = discovery.find_by_capability("statistical_analysis")[0]
agent = registry.get(agent_id)
result = agent.analyze([1, 2, 3], "mean")
print(result)
```

### 示例3：自动发现

```python
# 项目结构
# agents/
#   __init__.py
#   search_agent.py
#   analysis_agent.py

# search_agent.py
class SearchAgent:
    _agent_metadata = AgentMetadata(
        agent_id="search_auto",
        name="Auto Search Agent",
        description="Auto-discovered search agent",
        tags=["search"]
    )

# 自动发现
auto_registry = AutoDiscoveryRegistry()
auto_registry.auto_discover("agents")

# 列出所有Agent
all_agents = auto_registry.list_agents()
print(f"Discovered agents: {all_agents}")
```

### 示例4：高级查询

```python
registry = AgentRegistry()
advanced_discovery = AdvancedDiscovery(registry)

# 自定义过滤
def filter_by_version(metadata: AgentMetadata) -> bool:
    return metadata.version.startswith("2.")

agents_v2 = advanced_discovery.find_by_filter(filter_by_version)
print(f"Version 2 agents: {agents_v2}")

# 按schema查找
input_schema = {"query": "string"}
compatible_agents = advanced_discovery.find_by_schema(input_schema)
print(f"Compatible agents: {compatible_agents}")

# 按依赖查找
dependent_agents = advanced_discovery.find_by_dependency("database")
print(f"Agents needing database: {dependent_agents}")
```

### 示例5：动态管理

```python
dynamic_registry = DynamicRegistry()

# 注册Agent
metadata = AgentMetadata(
    agent_id="dynamic_001",
    name="Dynamic Agent",
    description="Dynamically managed agent"
)
agent = SearchAgent()
dynamic_registry.register("dynamic_001", agent, metadata)

# 激活Agent
dynamic_registry.activate("dynamic_001")

# 获取活跃Agent
active = dynamic_registry.get_active_agents()
print(f"Active agents: {active}")

# 停用Agent
dynamic_registry.deactivate("dynamic_001")

# 标记为弃用
dynamic_registry.deprecate("dynamic_001")
```

---

## 最佳实践

### 1. 元数据完整性

```python
# ✅ 好的做法：完整的元数据
metadata = AgentMetadata(
    agent_id="agent_001",
    name="Search Agent",
    description="Searches web and documents",
    version="2.1.0",
    tags=["search", "research", "web"],
    capabilities=["web_search", "document_search", "cache"],
    input_schema={"query": "string", "max_results": "integer"},
    output_schema={"results": "array", "total": "integer"},
    dependencies=["elasticsearch", "redis"]
)

# ❌ 避免：不完整的元数据
metadata = AgentMetadata(
    agent_id="agent_001",
    name="Agent"
)
```

### 2. 命名规范

```python
# ✅ 好的做法：清晰的命名
agent_ids = [
    "search_web_001",
    "analysis_statistical_001",
    "summary_text_001"
]

# ❌ 避免：模糊的命名
agent_ids = [
    "agent1",
    "agent2",
    "agent3"
]
```

### 3. 版本管理

```python
# ✅ 好的做法：语义版本
versions = ["1.0.0", "1.1.0", "2.0.0"]

# 支持版本查询
def find_by_version(registry, agent_name, version_range):
    agents = registry.list_metadata()
    return [
        agent_id for agent_id, metadata in agents.items()
        if metadata.name == agent_name and 
        metadata.version.startswith(version_range)
    ]
```

### 4. 缓存策略

```python
# ✅ 好的做法：合理的缓存
cached_discovery = CachedDiscovery(registry)

# 频繁查询使用缓存
for _ in range(1000):
    agents = cached_discovery.find_by_tag("search")

# 注册新Agent时清空缓存
registry.register("new_agent", agent, metadata)
cached_discovery.invalidate_cache()
```

### 5. 错误处理

```python
# ✅ 好的做法：完善的错误处理
def safe_find_agent(discovery, agent_id):
    try:
        agent = discovery.find_by_id(agent_id)
        if agent is None:
            raise ValueError(f"Agent {agent_id} not found")
        return agent
    except Exception as e:
        logger.error(f"Failed to find agent: {e}")
        return None
```

---

## 总结

LangGraph Agent注册与发现机制提供了：

- **灵活的注册**：支持手动、装饰器、自动发现等多种方式
- **高效的查询**：支持ID、标签、能力、schema等多维度查询
- **可扩展性**：支持中央、分布式、动态等多种架构
- **性能优化**：缓存、索引等优化技术
- **生命周期管理**：支持激活、停用、弃用等状态管理

通过合理使用这些机制，可以构建高效、可维护的多Agent系统。
