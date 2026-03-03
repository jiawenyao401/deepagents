"""技能中间件，用于加载并向系统提示暴露 Agent 技能。

本模块实现了 Anthropic 的 Agent 技能模式（渐进式披露），
从后端存储通过可配置的来源加载技能。

## Architecture

Skills are loaded from one or more **sources** - paths in a backend where skills are
organized. Sources are loaded in order, with later sources overriding earlier ones
when skills have the same name (last one wins). This enables layering: base -> user
-> project -> team skills.

The middleware uses backend APIs exclusively (no direct filesystem access), making it
portable across different storage backends (filesystem, state, remote storage, etc.).

For StateBackend (ephemeral/in-memory), use a factory function:
```python
SkillsMiddleware(backend=lambda rt: StateBackend(rt), ...)
```

## Skill Structure

Each skill is a directory containing a SKILL.md file with YAML frontmatter:

```
/skills/user/web-research/
├── SKILL.md          # Required: YAML frontmatter + markdown instructions
└── helper.py         # Optional: supporting files
```

SKILL.md format:
```markdown
---
name: web-research
description: Structured approach to conducting thorough web research
license: MIT
---

# Web Research Skill

## When to Use
- User asks you to research a topic
...
```

## Skill Metadata (SkillMetadata)

Parsed from YAML frontmatter per Agent Skills specification:
- `name`: Skill identifier (max 64 chars, lowercase alphanumeric and hyphens)
- `description`: What the skill does (max 1024 chars)
- `path`: Backend path to the SKILL.md file
- Optional: `license`, `compatibility`, `metadata`, `allowed_tools`

## Sources

Sources are simply paths to skill directories in the backend. The source name is
derived from the last component of the path (e.g., "/skills/user/" -> "user").

Example sources:
```python
[
    "/skills/user/",
    "/skills/project/"
]
```

## Path Conventions

All paths use POSIX conventions (forward slashes) via `PurePosixPath`:
- Backend paths: "/skills/user/web-research/SKILL.md"
- Virtual, platform-independent
- Backends handle platform-specific conversions as needed

## Usage

```python
from deepagents.backends.state import StateBackend
from deepagents.middleware.skills import SkillsMiddleware

middleware = SkillsMiddleware(
    backend=my_backend,
    sources=[
        "/skills/base/",
        "/skills/user/",
        "/skills/project/",
    ],
)
```
"""

from __future__ import annotations

import logging
import re
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Annotated

import yaml
from langchain.agents.middleware.types import PrivateStateAttr

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain_core.runnables import RunnableConfig
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

from typing import NotRequired, TypedDict

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langgraph.prebuilt import ToolRuntime

from deepagents.middleware._utils import append_to_system_message

logger = logging.getLogger(__name__)

# 安全限制：SKILL.md 文件最大尺寸，防止 DoS 攻击（10MB）
MAX_SKILL_FILE_SIZE = 10 * 1024 * 1024

# Agent Skills 规范约束（https://agentskills.io/specification）
MAX_SKILL_NAME_LENGTH = 64          # 技能名称最大长度
MAX_SKILL_DESCRIPTION_LENGTH = 1024  # 技能描述最大长度
MAX_SKILL_COMPATIBILITY_LENGTH = 500 # 兼容性字段最大长度


class SkillMetadata(TypedDict):
    """技能元数据，遵循 Agent Skills 规范（https://agentskills.io/specification）。"""

    path: str
    """SKILL.md 文件的路径。"""

    name: str
    """技能标识符。

    Agent Skills 规范约束：
    - 1-64 个字符
    - 仅允许 Unicode 小写字母、数字和连字符（a-z 和 -）
    - 不能以 - 开头或结尾
    - 不能包含连续的 --
    - 必须与包含 SKILL.md 文件的父目录名称一致
    """

    description: str
    """技能的功能描述。

    Agent Skills 规范约束：
    - 1-1024 个字符
    - 应同时描述技能的功能和使用时机
    - 应包含帮助 Agent 识别相关任务的关键词
    """

    license: str | None
    """许可证名称或对捆绑许可证文件的引用。"""

    compatibility: str | None
    """环境要求说明。

    Agent Skills 规范约束：
    - 若提供，长度为 1-500 个字符
    - 仅在有特定兼容性要求时才包含
    - 可指明目标产品、所需包等
    """

    metadata: dict[str, str]
    """附加元数据的任意键值映射。

    客户端可用此字段存储规范未定义的额外属性。
    建议保持键名唯一以避免冲突。
    """

    allowed_tools: list[str]
    """技能推荐使用的工具名称列表。

    警告：此字段为实验性功能。
    Agent Skills 规范约束：以空格分隔的工具名称列表。
    """


class SkillsState(AgentState):
    """技能中间件的状态模式。"""

    # 私有字段：存储已加载的技能元数据列表，不会传播到父 Agent
    skills_metadata: NotRequired[Annotated[list[SkillMetadata], PrivateStateAttr]]
    """从配置来源加载的技能元数据列表，不会传播到父 Agent。"""


class SkillsStateUpdate(TypedDict):
    """技能中间件的状态更新结构。"""

    skills_metadata: list[SkillMetadata]
    """待合并到状态中的技能元数据列表。"""


def _validate_skill_name(name: str, directory_name: str) -> tuple[bool, str]:
    """按 Agent Skills 规范校验技能名称。

    规范约束：
    - 1-64 个字符
    - 仅允许 Unicode 小写字母、数字和连字符（a-z 和 -）
    - 不能以 - 开头或结尾
    - 不能包含连续的 --
    - 必须与包含 SKILL.md 文件的父目录名称一致

    Unicode 小写字母数字指满足 `c.isalpha() and c.islower()` 或 `c.isdigit()` 的字符，
    涵盖带重音的拉丁字符（如 'café'、'über-tool'）及其他文字。

    Args:
        name: YAML frontmatter 中的技能名称。
        directory_name: 包含 SKILL.md 的父目录名称。

    Returns:
        `(is_valid, error_message)` 元组，校验通过时 error_message 为空字符串。
    """
    if not name:
        return False, "name is required"
    if len(name) > MAX_SKILL_NAME_LENGTH:
        return False, "name exceeds 64 characters"
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False, "name must be lowercase alphanumeric with single hyphens only"
    for c in name:
        if c == "-":
            continue
        if (c.isalpha() and c.islower()) or c.isdigit():
            continue
        return False, "name must be lowercase alphanumeric with single hyphens only"
    if name != directory_name:
        return False, f"name '{name}' must match directory name '{directory_name}'"
    return True, ""


def _parse_skill_metadata(  # noqa: C901
    content: str,
    skill_path: str,
    directory_name: str,
) -> SkillMetadata | None:
    """从 SKILL.md 内容中解析 YAML frontmatter 元数据。

    按 Agent Skills 规范从文件开头的 `---` 分隔符之间提取 YAML frontmatter。

    Args:
        content: SKILL.md 文件的完整内容。
        skill_path: SKILL.md 文件路径（用于错误信息和元数据）。
        directory_name: 包含该技能的父目录名称。

    Returns:
        解析成功返回 SkillMetadata，解析失败或校验不通过返回 None。
    """
    if len(content) > MAX_SKILL_FILE_SIZE:
        logger.warning("Skipping %s: content too large (%d bytes)", skill_path, len(content))
        return None

    # Match YAML frontmatter between --- delimiters
    frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n"
    match = re.match(frontmatter_pattern, content, re.DOTALL)

    if not match:
        logger.warning("Skipping %s: no valid YAML frontmatter found", skill_path)
        return None

    frontmatter_str = match.group(1)

    # Parse YAML using safe_load for proper nested structure support
    try:
        frontmatter_data = yaml.safe_load(frontmatter_str)
    except yaml.YAMLError as e:
        logger.warning("Invalid YAML in %s: %s", skill_path, e)
        return None

    if not isinstance(frontmatter_data, dict):
        logger.warning("Skipping %s: frontmatter is not a mapping", skill_path)
        return None

    name = str(frontmatter_data.get("name", "")).strip()
    description = str(frontmatter_data.get("description", "")).strip()
    if not name or not description:
        logger.warning("Skipping %s: missing required 'name' or 'description'", skill_path)
        return None

    # Validate name format per spec (warn but continue loading for backwards compatibility)
    is_valid, error = _validate_skill_name(str(name), directory_name)
    if not is_valid:
        logger.warning(
            "Skill '%s' in %s does not follow Agent Skills specification: %s. Consider renaming for spec compliance.",
            name,
            skill_path,
            error,
        )

    description_str = description
    if len(description_str) > MAX_SKILL_DESCRIPTION_LENGTH:
        logger.warning(
            "Description exceeds %d characters in %s, truncating",
            MAX_SKILL_DESCRIPTION_LENGTH,
            skill_path,
        )
        description_str = description_str[:MAX_SKILL_DESCRIPTION_LENGTH]

    raw_tools = frontmatter_data.get("allowed-tools")
    if isinstance(raw_tools, str):
        allowed_tools = [
            t.strip(",")  # Support commas for compatibility with skills created for Claude Code.
            for t in raw_tools.split()
            if t.strip(",")
        ]
    else:
        if raw_tools is not None:
            logger.warning(
                "Ignoring non-string 'allowed-tools' in %s (got %s)",
                skill_path,
                type(raw_tools).__name__,
            )
        allowed_tools = []

    compatibility_str = str(frontmatter_data.get("compatibility", "")).strip() or None
    if compatibility_str and len(compatibility_str) > MAX_SKILL_COMPATIBILITY_LENGTH:
        logger.warning(
            "Compatibility exceeds %d characters in %s, truncating",
            MAX_SKILL_COMPATIBILITY_LENGTH,
            skill_path,
        )
        compatibility_str = compatibility_str[:MAX_SKILL_COMPATIBILITY_LENGTH]

    return SkillMetadata(
        name=str(name),
        description=description_str,
        path=skill_path,
        metadata=_validate_metadata(frontmatter_data.get("metadata", {}), skill_path),
        license=str(frontmatter_data.get("license", "")).strip() or None,
        compatibility=compatibility_str,
        allowed_tools=allowed_tools,
    )


def _validate_metadata(
    raw: object,
    skill_path: str,
) -> dict[str, str]:
    """校验并规范化 YAML frontmatter 中的 metadata 字段。

    YAML safe_load 对 metadata 键可能返回任意类型。
    本函数通过 str() 强制转换并拒绝非字典输入，
    确保 SkillMetadata 中的值始终为 dict[str, str]。

    Args:
        raw: frontmatter_data.get("metadata", {}) 的原始值。
        skill_path: SKILL.md 文件路径（用于警告信息）。

    Returns:
        校验后的 dict[str, str]。
    """
    if not isinstance(raw, dict):
        if raw:
            logger.warning(
                "Ignoring non-dict metadata in %s (got %s)",
                skill_path,
                type(raw).__name__,
            )
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def _format_skill_annotations(skill: SkillMetadata) -> str:
    """从技能的可选字段构建括号注释字符串。

    将 license 和 compatibility 合并为逗号分隔的字符串，
    用于在系统提示的技能列表中显示。

    Args:
        skill: 待提取注释的技能元数据。

    Returns:
        形如 'License: MIT, Compatibility: Python 3.10+' 的注释字符串，
        若两个字段均未设置则返回空字符串。
    """
    parts: list[str] = []
    if skill.get("license"):
        parts.append(f"License: {skill['license']}")
    if skill.get("compatibility"):
        parts.append(f"Compatibility: {skill['compatibility']}")
    return ", ".join(parts)


def _list_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """从后端来源列出所有技能（同步版本）。

    扫描后端中包含 SKILL.md 文件的子目录，下载内容，
    解析 YAML frontmatter，并返回技能元数据列表。

    预期目录结构：
    ```txt
    source_path/
    └── skill-name/
        ├── SKILL.md   # 必须
        └── helper.py  # 可选
    ```

    Args:
        backend: 用于文件操作的后端实例。
        source_path: 后端中技能目录的路径。

    Returns:
        成功解析的 SKILL.md 文件对应的技能元数据列表。
    """
    skills: list[SkillMetadata] = []
    items = backend.ls_info(source_path)

    # Find all skill directories (directories containing SKILL.md)
    skill_dirs = []
    for item in items:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # For each skill directory, check if SKILL.md exists and download it
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # Construct SKILL.md path using PurePosixPath for safe, standardized path operations
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = backend.download_files(paths_to_download)

    # Parse each downloaded SKILL.md
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # Skill doesn't have a SKILL.md, skip it
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # Extract directory name from path using PurePosixPath
        directory_name = PurePosixPath(skill_dir_path).name

        # Parse metadata
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


async def _alist_skills(backend: BackendProtocol, source_path: str) -> list[SkillMetadata]:
    """从后端来源列出所有技能（异步版本）。

    扫描后端中包含 SKILL.md 文件的子目录，异步下载内容，
    解析 YAML frontmatter，并返回技能元数据列表。

    预期目录结构：
    ```txt
    source_path/
    └── skill-name/
        ├── SKILL.md   # 必须
        └── helper.py  # 可选
    ```

    Args:
        backend: 用于文件操作的后端实例。
        source_path: 后端中技能目录的路径。

    Returns:
        成功解析的 SKILL.md 文件对应的技能元数据列表。
    """
    skills: list[SkillMetadata] = []
    items = await backend.als_info(source_path)

    # Find all skill directories (directories containing SKILL.md)
    skill_dirs = []
    for item in items:
        if not item.get("is_dir"):
            continue
        skill_dirs.append(item["path"])

    if not skill_dirs:
        return []

    # For each skill directory, check if SKILL.md exists and download it
    skill_md_paths = []
    for skill_dir_path in skill_dirs:
        # Construct SKILL.md path using PurePosixPath for safe, standardized path operations
        skill_dir = PurePosixPath(skill_dir_path)
        skill_md_path = str(skill_dir / "SKILL.md")
        skill_md_paths.append((skill_dir_path, skill_md_path))

    paths_to_download = [skill_md_path for _, skill_md_path in skill_md_paths]
    responses = await backend.adownload_files(paths_to_download)

    # Parse each downloaded SKILL.md
    for (skill_dir_path, skill_md_path), response in zip(skill_md_paths, responses, strict=True):
        if response.error:
            # Skill doesn't have a SKILL.md, skip it
            continue

        if response.content is None:
            logger.warning("Downloaded skill file %s has no content", skill_md_path)
            continue

        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError as e:
            logger.warning("Error decoding %s: %s", skill_md_path, e)
            continue

        # Extract directory name from path using PurePosixPath
        directory_name = PurePosixPath(skill_dir_path).name

        # Parse metadata
        skill_metadata = _parse_skill_metadata(
            content=content,
            skill_path=skill_md_path,
            directory_name=directory_name,
        )
        if skill_metadata:
            skills.append(skill_metadata)

    return skills


SKILLS_SYSTEM_PROMPT = """

## Skills System

You have access to a skills library that provides specialized capabilities and domain knowledge.

{skills_locations}

**Available Skills:**

{skills_list}

**How to Use Skills (Progressive Disclosure):**

Skills follow a **progressive disclosure** pattern - you see their name and description above, but only read full instructions when needed:

1. **Recognize when a skill applies**: Check if the user's task matches a skill's description
2. **Read the skill's full instructions**: Use the path shown in the skill list above
3. **Follow the skill's instructions**: SKILL.md contains step-by-step workflows, best practices, and examples
4. **Access supporting files**: Skills may include helper scripts, configs, or reference docs - use absolute paths

**When to Use Skills:**
- User's request matches a skill's domain (e.g., "research X" -> web-research skill)
- You need specialized knowledge or structured workflows
- A skill provides proven patterns for complex tasks

**Executing Skill Scripts:**
Skills may contain Python scripts or other executable files. Always use absolute paths from the skill list.

**Example Workflow:**

User: "Can you research the latest developments in quantum computing?"

1. Check available skills -> See "web-research" skill with its path
2. Read the skill using the path shown
3. Follow the skill's research workflow (search -> organize -> synthesize)
4. Use any helper scripts with absolute paths

Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!
"""


class SkillsMiddleware(AgentMiddleware[SkillsState, ContextT, ResponseT]):
    """Middleware for loading and exposing agent skills to the system prompt.

    Loads skills from backend sources and injects them into the system prompt
    using progressive disclosure (metadata first, full content on demand).

    Skills are loaded in source order with later sources overriding
    earlier ones.

    Example:
        ```python
        from deepagents.backends.filesystem import FilesystemBackend

        backend = FilesystemBackend(root_dir="/path/to/skills")
        middleware = SkillsMiddleware(
            backend=backend,
            sources=[
                "/path/to/skills/user/",
                "/path/to/skills/project/",
            ],
        )
        ```

    Args:
        backend: Backend instance for file operations
        sources: List of skill source paths.

            Source names are derived from the last path component.
    """

    state_schema = SkillsState

    def __init__(self, *, backend: BACKEND_TYPES, sources: list[str]) -> None:
        """Initialize the skills middleware.

        Args:
            backend: Backend instance or factory function that takes runtime and
                returns a backend.

                Use a factory for StateBackend: `lambda rt: StateBackend(rt)`
            sources: List of skill source paths (e.g.,
                `['/skills/user/', '/skills/project/']`).
        """
        self._backend = backend
        self.sources = sources
        self.system_prompt_template = SKILLS_SYSTEM_PROMPT

    def _get_backend(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> BackendProtocol:
        """从实例或工厂函数解析后端。

        若 _backend 是可调用对象，则构造临时 ToolRuntime 调用工厂函数；
        否则直接返回后端实例。

        Args:
            state: 当前 Agent 状态。
            runtime: 工厂函数所需的运行时上下文。
            config: 传递给后端工厂的可运行配置。

        Returns:
            解析后的后端实例。
        """
        if callable(self._backend):
            # Construct an artificial tool runtime to resolve backend factory
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            backend = self._backend(tool_runtime)  # ty: ignore[call-top-callable, invalid-argument-type]
            if backend is None:
                msg = "SkillsMiddleware requires a valid backend instance"
                raise AssertionError(msg)
            return backend

        return self._backend

    def _format_skills_locations(self) -> str:
        """格式化技能来源路径，用于在系统提示中显示。"""
        locations = []

        for i, source_path in enumerate(self.sources):
            name = PurePosixPath(source_path.rstrip("/")).name.capitalize()
            suffix = " (higher priority)" if i == len(self.sources) - 1 else ""
            locations.append(f"**{name} Skills**: `{source_path}`{suffix}")

        return "\n".join(locations)

    def _format_skills_list(self, skills: list[SkillMetadata]) -> str:
        """格式化技能元数据列表，用于在系统提示中显示。"""
        if not skills:
            paths = [f"{source_path}" for source_path in self.sources]
            return f"(No skills available yet. You can create skills in {' or '.join(paths)})"

        lines = []
        for skill in skills:
            annotations = _format_skill_annotations(skill)
            desc_line = f"- **{skill['name']}**: {skill['description']}"
            if annotations:
                desc_line += f" ({annotations})"
            lines.append(desc_line)
            if skill["allowed_tools"]:
                lines.append(f"  -> Allowed tools: {', '.join(skill['allowed_tools'])}")
            lines.append(f"  -> Read `{skill['path']}` for full instructions")

        return "\n".join(lines)

    def modify_request(self, request: ModelRequest[ContextT]) -> ModelRequest[ContextT]:
        """将技能文档注入模型请求的系统消息。

        Args:
            request: 待修改的模型请求。

        Returns:
            注入了技能文档的新模型请求。
        """
        skills_metadata = request.state.get("skills_metadata", [])
        skills_locations = self._format_skills_locations()
        skills_list = self._format_skills_list(skills_metadata)

        skills_section = self.system_prompt_template.format(
            skills_locations=skills_locations,
            skills_list=skills_list,
        )

        new_system_message = append_to_system_message(request.system_message, skills_section)

        return request.override(system_message=new_system_message)

    def before_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        """在 Agent 执行前同步加载技能元数据。

        每次会话只加载一次。若状态中已存在 skills_metadata（来自上一轮或检查点），
        则跳过加载并返回 None。

        按来源顺序加载，同名技能后来源覆盖前来源（后者优先）。

        Args:
            state: 当前 Agent 状态。
            runtime: 运行时上下文。
            config: 可运行配置。

        Returns:
            包含已填充 skills_metadata 的状态更新，若已存在则返回 None。
        """
        # Skip if skills_metadata is already present in state (even if empty)
        if "skills_metadata" in state:
            return None

        # Resolve backend (supports both direct instances and factory functions)
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}

        # Load skills from each source in order
        # Later sources override earlier ones (last one wins)
        for source_path in self.sources:
            source_skills = _list_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        return SkillsStateUpdate(skills_metadata=skills)

    async def abefore_agent(self, state: SkillsState, runtime: Runtime, config: RunnableConfig) -> SkillsStateUpdate | None:  # ty: ignore[invalid-method-override]
        """在 Agent 执行前异步加载技能元数据。

        每次会话只加载一次。若状态中已存在 skills_metadata（来自上一轮或检查点），
        则跳过加载并返回 None。

        按来源顺序加载，同名技能后来源覆盖前来源（后者优先）。

        Args:
            state: 当前 Agent 状态。
            runtime: 运行时上下文。
            config: 可运行配置。

        Returns:
            包含已填充 skills_metadata 的状态更新，若已存在则返回 None。
        """
        # Skip if skills_metadata is already present in state (even if empty)
        if "skills_metadata" in state:
            return None

        # Resolve backend (supports both direct instances and factory functions)
        backend = self._get_backend(state, runtime, config)
        all_skills: dict[str, SkillMetadata] = {}

        # Load skills from each source in order
        # Later sources override earlier ones (last one wins)
        for source_path in self.sources:
            source_skills = await _alist_skills(backend, source_path)
            for skill in source_skills:
                all_skills[skill["name"]] = skill

        skills = list(all_skills.values())
        return SkillsStateUpdate(skills_metadata=skills)

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT]:
        """将技能文档注入系统提示（同步版本）。

        Args:
            request: 正在处理的模型请求。
            handler: 使用修改后请求调用的处理函数。

        Returns:
            处理函数返回的模型响应。
        """
        modified_request = self.modify_request(request)
        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT]:
        """将技能文档注入系统提示（异步版本）。

        Args:
            request: 正在处理的模型请求。
            handler: 使用修改后请求调用的异步处理函数。

        Returns:
            处理函数返回的模型响应。
        """
        modified_request = self.modify_request(request)
        return await handler(modified_request)


__all__ = ["SkillMetadata", "SkillsMiddleware"]
