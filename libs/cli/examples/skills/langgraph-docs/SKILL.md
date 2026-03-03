---
name: langgraph-docs
description: Use this skill for requests related to LangGraph in order to fetch relevant documentation to provide accurate, up-to-date guidance.
---

# langgraph-docs

## Overview

This skill explains how to access LangGraph Python documentation to help answer questions and guide implementation.
（该技能解释了如何访问LangGraph Python文档，以帮助解答问题并指导实施。）
## Instructions
（说明书）

### 1. Fetch the Documentation Index
（获取文档索引）

Use the fetch_url tool to read the following URL:
（使用fetch_url工具阅读以下网址：）
https://docs.langchain.com/llms.txt

This provides a structured list of all available documentation with descriptions.
（它提供了所有可用文档的结构化列表及描述。）
### 2. Select Relevant Documentation
（选择相关文献）

Based on the question, identify 2-4 most relevant documentation URLs from the index. Prioritize:
- Specific how-to guides for implementation questions
- Core concept pages for understanding questions
- Tutorials for end-to-end examples
- Reference docs for API details
（根据问题，从索引中找出2到4个最相关的文档URL。
- 优先排序：-
- 实现问题的具体作指南 
- - 理解问题的核心概念页面 
- - 端到端示例教程 
- - API 详情参考文档）
### 3. Fetch Selected Documentation
（获取精选文档）
Use the fetch_url tool to read the selected documentation URLs.
（使用fetch_url工具阅读选定的文档URL。）

### 4. Provide Accurate Guidance
（提供准确的指导）

After reading the documentation, complete the users request.
（阅读文档后，完成用户请求。）
