# Python typing 类型注解详解

## 目录
1. [基础类型](#基础类型)
2. [Any类型](#any类型)
3. [Callable类型](#callable类型)
4. [Optional类型](#optional类型)
5. [其他常用类型](#其他常用类型)
6. [应用场景对比](#应用场景对比)
7. [最佳实践](#最佳实践)

---

## 基础类型

### Python内置类型注解

```python
# 基础类型
x: int = 10
y: str = "hello"
z: float = 3.14
flag: bool = True

# 容器类型
numbers: list = [1, 2, 3]
mapping: dict = {"key": "value"}
items: tuple = (1, "two", 3.0)
unique: set = {1, 2, 3}

# 函数返回类型
def greet(name: str) -> str:
    return f"Hello, {name}"
```

---

## Any类型

### 定义

`Any` 是一个特殊的类型，表示**任意类型**。它禁用类型检查，允许任何值。

```python
from typing import Any

# Any可以接收任何类型的值
value: Any = 10
value = "string"
value = [1, 2, 3]
value = {"key": "value"}
value = None
# 都不会产生类型错误
```

### 特点

| 特点 | 说明 |
|------|------|
| **最宽松** | 接受任何类型的值 |
| **禁用检查** | 类型检查器不会报错 |
| **灵活性高** | 适合处理未知类型 |
| **可读性差** | 不能清楚表达意图 |

### 应用场景

```python
# 场景1：处理未知类型的数据
def process_data(data: Any) -> Any:
    """处理任意类型的数据"""
    if isinstance(data, str):
        return data.upper()
    elif isinstance(data, int):
        return data * 2
    elif isinstance(data, list):
        return len(data)
    return data

# 场景2：与动态库交互
def call_external_api(params: Any) -> Any:
    """调用外部API，参数和返回值类型不确定"""
    import requests
    response = requests.post("https://api.example.com", json=params)
    return response.json()

# 场景3：通用容器
def get_first_element(container: Any) -> Any:
    """获取容器的第一个元素"""
    try:
        return container[0]
    except (IndexError, TypeError):
        return None

# 场景4：配置字典
config: dict[str, Any] = {
    "name": "app",
    "port": 8080,
    "debug": True,
    "features": ["auth", "logging"],
    "timeout": 30.5
}
```

### 何时使用Any

```python
# ✅ 合理使用Any
def deserialize(json_str: str) -> Any:
    """JSON反序列化，返回类型不确定"""
    import json
    return json.loads(json_str)

# ❌ 避免过度使用Any
def add(a: Any, b: Any) -> Any:  # 不好
    return a + b

# ✅ 更好的做法
def add(a: int | float, b: int | float) -> int | float:
    return a + b
```

---

## Callable类型

### 定义

`Callable` 表示**可调用对象**（函数、方法、类等）。它定义了函数的参数类型和返回类型。

```python
from typing import Callable

# 基础语法：Callable[[参数类型], 返回类型]

# 无参数函数
func1: Callable[[], int] = lambda: 42

# 单参数函数
func2: Callable[[str], int] = len

# 多参数函数
func3: Callable[[int, int], int] = lambda x, y: x + y

# 可变参数函数
func4: Callable[..., str] = print  # ... 表示任意参数
```

### 详细示例

```python
from typing import Callable

# 示例1：回调函数
def process_with_callback(
    data: list[int],
    callback: Callable[[int], int]
) -> list[int]:
    """对每个元素应用回调函数"""
    return [callback(x) for x in data]

# 使用
result = process_with_callback([1, 2, 3], lambda x: x * 2)
print(result)  # [2, 4, 6]

# 示例2：函数工厂
def create_multiplier(factor: int) -> Callable[[int], int]:
    """创建乘法函数"""
    def multiplier(x: int) -> int:
        return x * factor
    return multiplier

multiply_by_3 = create_multiplier(3)
print(multiply_by_3(5))  # 15

# 示例3：装饰器
def timer(func: Callable[..., Any]) -> Callable[..., Any]:
    """计时装饰器"""
    import time
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"Execution time: {time.time() - start}s")
        return result
    return wrapper

@timer
def slow_function():
    import time
    time.sleep(1)

# 示例4：事件处理
class EventEmitter:
    def __init__(self):
        self.handlers: dict[str, list[Callable]] = {}
    
    def on(self, event: str, handler: Callable[[Any], None]):
        """注册事件处理器"""
        if event not in self.handlers:
            self.handlers[event] = []
        self.handlers[event].append(handler)
    
    def emit(self, event: str, data: Any):
        """触发事件"""
        if event in self.handlers:
            for handler in self.handlers[event]:
                handler(data)

# 使用
emitter = EventEmitter()
emitter.on("click", lambda data: print(f"Clicked: {data}"))
emitter.emit("click", "button1")
```

### 应用场景

```python
# 场景1：高阶函数
def map_function(
    func: Callable[[int], int],
    items: list[int]
) -> list[int]:
    """映射函数"""
    return [func(item) for item in items]

# 场景2：策略模式
class DataProcessor:
    def __init__(self, strategy: Callable[[list], list]):
        self.strategy = strategy
    
    def process(self, data: list) -> list:
        return self.strategy(data)

# 场景3：异步回调
async def fetch_data(
    url: str,
    on_success: Callable[[dict], None],
    on_error: Callable[[Exception], None]
):
    """异步获取数据"""
    try:
        # 模拟网络请求
        data = {"status": "ok"}
        on_success(data)
    except Exception as e:
        on_error(e)

# 场景4：排序和过滤
numbers = [3, 1, 4, 1, 5, 9]
sorted_nums = sorted(numbers, key=lambda x: -x)  # Callable[[int], int]
filtered = filter(lambda x: x > 3, numbers)  # Callable[[int], bool]
```

---

## Optional类型

### 定义

`Optional[T]` 表示值可以是类型 `T` 或 `None`。它是 `T | None` 的简写。

```python
from typing import Optional

# 这两种写法等价
x: Optional[int] = None
y: int | None = None

# 都表示可以是int或None
value1: Optional[str] = "hello"
value2: Optional[str] = None
```

### 详细示例

```python
from typing import Optional

# 示例1：可选参数
def greet(name: Optional[str] = None) -> str:
    """问候函数，名字可选"""
    if name is None:
        return "Hello, stranger!"
    return f"Hello, {name}!"

print(greet())  # Hello, stranger!
print(greet("Alice"))  # Hello, Alice!

# 示例2：可选返回值
def find_user(user_id: int) -> Optional[dict]:
    """查找用户，可能不存在"""
    users = {1: {"name": "Alice"}, 2: {"name": "Bob"}}
    return users.get(user_id)

user = find_user(1)
if user is not None:
    print(user["name"])

# 示例3：可选属性
class Config:
    def __init__(self, name: str, timeout: Optional[int] = None):
        self.name = name
        self.timeout = timeout or 30  # 默认值

config = Config("app")
print(config.timeout)  # 30

# 示例4：链式调用
class User:
    def __init__(self, name: str):
        self.name = name
        self.profile: Optional["Profile"] = None

class Profile:
    def __init__(self, bio: str):
        self.bio = bio

def get_user_bio(user: Optional[User]) -> Optional[str]:
    """获取用户简介"""
    if user is None:
        return None
    if user.profile is None:
        return None
    return user.profile.bio

# 使用
user = User("Alice")
bio = get_user_bio(user)  # None
```

### 应用场景

```python
# 场景1：数据库查询
def query_by_id(table: str, id: int) -> Optional[dict]:
    """查询数据库，可能返回None"""
    # 模拟数据库查询
    return None if id < 0 else {"id": id, "data": "..."}

# 场景2：配置参数
def create_connection(
    host: str,
    port: int,
    username: Optional[str] = None,
    password: Optional[str] = None
) -> str:
    """创建连接"""
    auth = ""
    if username and password:
        auth = f"{username}:{password}@"
    return f"Connection: {auth}{host}:{port}"

# 场景3：缓存
cache: dict[str, Optional[str]] = {}

def get_cached_value(key: str) -> Optional[str]:
    """获取缓存值"""
    return cache.get(key)

# 场景4：错误处理
def safe_divide(a: float, b: float) -> Optional[float]:
    """安全除法"""
    if b == 0:
        return None
    return a / b
```

---

## 其他常用类型

### Union类型

```python
from typing import Union

# Union表示多个可能的类型
value: Union[int, str] = 10
value = "hello"

# Python 3.10+ 可以用 | 替代
value: int | str = 10

# 多个类型
result: int | str | float | None = None
```

### List、Dict、Tuple

```python
from typing import List, Dict, Tuple

# 旧写法（Python 3.8及以前）
numbers: List[int] = [1, 2, 3]
mapping: Dict[str, int] = {"a": 1, "b": 2}
pair: Tuple[str, int] = ("name", 25)

# 新写法（Python 3.9+）
numbers: list[int] = [1, 2, 3]
mapping: dict[str, int] = {"a": 1, "b": 2}
pair: tuple[str, int] = ("name", 25)
```

### TypeVar和Generic

```python
from typing import TypeVar, Generic

T = TypeVar('T')  # 泛型类型变量

class Container(Generic[T]):
    def __init__(self, value: T):
        self.value = value
    
    def get(self) -> T:
        return self.value

# 使用
int_container: Container[int] = Container(42)
str_container: Container[str] = Container("hello")
```

---

## 应用场景对比

### 场景1：API响应处理

```python
from typing import Any, Optional, Union

# ❌ 不好：使用Any
def parse_response(response: Any) -> Any:
    return response.get("data")

# ✅ 好：明确类型
def parse_response(response: dict[str, Any]) -> Optional[dict]:
    return response.get("data")

# ✅ 更好：完整类型
def parse_response(response: dict[str, Any]) -> dict[str, Any] | None:
    return response.get("data")
```

### 场景2：回调处理

```python
from typing import Callable, Optional

# ❌ 不好：使用Any
def process_async(data: Any, callback: Any):
    result = transform(data)
    callback(result)

# ✅ 好：明确类型
def process_async(
    data: dict,
    callback: Callable[[dict], None]
):
    result = transform(data)
    callback(result)

# ✅ 更好：支持可选回调
def process_async(
    data: dict,
    callback: Optional[Callable[[dict], None]] = None
):
    result = transform(data)
    if callback:
        callback(result)
    return result
```

### 场景3：配置管理

```python
from typing import Any, Optional

# ❌ 不好：过度使用Any
config: dict[str, Any] = {
    "name": "app",
    "port": 8080,
    "debug": True
}

# ✅ 好：使用TypedDict
from typing import TypedDict

class AppConfig(TypedDict):
    name: str
    port: int
    debug: bool
    timeout: Optional[int]

config: AppConfig = {
    "name": "app",
    "port": 8080,
    "debug": True,
    "timeout": None
}
```

### 场景4：工具函数

```python
from typing import Callable, Optional, TypeVar

T = TypeVar('T')

# ❌ 不好：使用Any
def retry(func: Any, times: int = 3) -> Any:
    for _ in range(times):
        try:
            return func()
        except:
            pass
    return None

# ✅ 好：使用泛型
def retry(
    func: Callable[[], T],
    times: int = 3
) -> Optional[T]:
    for _ in range(times):
        try:
            return func()
        except:
            pass
    return None

# 使用
result: Optional[int] = retry(lambda: 42)
```

---

## 最佳实践

### 1. 优先级顺序

```python
# 优先级：具体类型 > Union > Optional > Any

# 1️⃣ 最好：具体类型
def process(data: dict[str, int]) -> list[int]:
    return list(data.values())

# 2️⃣ 次好：Union
def process(data: dict | list) -> int:
    if isinstance(data, dict):
        return len(data)
    return len(data)

# 3️⃣ 可接受：Optional
def process(data: Optional[dict]) -> Optional[int]:
    if data is None:
        return None
    return len(data)

# 4️⃣ 最后：Any
def process(data: Any) -> Any:
    return len(data)
```

### 2. 避免过度使用Any

```python
# ❌ 避免
def transform(data: Any) -> Any:
    return data.upper()

# ✅ 改进
def transform(data: str) -> str:
    return data.upper()

# ✅ 如果确实需要灵活性
def transform(data: str | bytes) -> str | bytes:
    if isinstance(data, bytes):
        return data.upper()
    return data.upper()
```

### 3. Callable的正确使用

```python
# ❌ 避免
def apply(func: Callable, data: list) -> list:
    return [func(x) for x in data]

# ✅ 改进
def apply(func: Callable[[int], int], data: list[int]) -> list[int]:
    return [func(x) for x in data]

# ✅ 更灵活
from typing import TypeVar

T = TypeVar('T')
U = TypeVar('U')

def apply(func: Callable[[T], U], data: list[T]) -> list[U]:
    return [func(x) for x in data]
```

### 4. Optional的正确使用

```python
# ❌ 避免
def get_value(key: str) -> Optional[Any]:
    return cache.get(key)

# ✅ 改进
def get_value(key: str) -> Optional[str]:
    return cache.get(key)

# ✅ 使用Union表示多种可能
def get_value(key: str) -> str | int | None:
    return cache.get(key)
```

### 5. 类型检查工具

```python
# 使用mypy进行类型检查
# mypy script.py

# 使用pyright
# pyright script.py

# 在代码中禁用检查
def legacy_function(data):  # type: ignore
    return data.process()
```

---

## 总结

| 类型 | 用途 | 灵活性 | 可读性 |
|------|------|--------|--------|
| **Any** | 任意类型 | 最高 | 最低 |
| **Callable** | 函数/可调用对象 | 中等 | 中等 |
| **Optional** | 可选值 | 中等 | 高 |
| **Union** | 多种类型 | 中等 | 中等 |
| **具体类型** | 明确类型 | 最低 | 最高 |

**建议：**
- 优先使用具体类型
- 必要时使用Union或Optional
- 避免过度使用Any
- 使用类型检查工具验证代码
