# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供本代码库的工作指引。

## 语言规则

**所有回答和显示都必须采用中文。**

- 所有输出、解释、注释、文档均使用中文
- 代码中的字符串、日志、错误信息保持中文
- 用户界面文本、提示信息使用中文

## 项目概述

DummyCode 是一个基于 Python 的 AI 智能代理项目，集成了大语言模型（LLM）、模型上下文协议（MCP）和检索增强生成（RAG）技术。支持多会话管理，每个会话拥有独立的记忆和知识图谱。

## 常用命令

### 运行应用

```bash
cd src
python main.py
```

### 安装依赖

```bash
# Python 依赖
pip install -r requirements.txt

# MCP 服务器（功能必需）
npm install -g @modelcontextprotocol/server-filesystem

# FAISS（用于 RAG 向量存储）- 推荐通过 conda 安装
conda install -c pytorch faiss-cpu
# 或通过 pip
pip install faiss-cpu
```

### 环境配置

复制 `.env.example` 为 `.env` 并配置：
- `OPENAI_API_KEY` / `OPENAI_BASE_URL`：大语言模型 API（默认配置为 DeepSeek）
- `EMBEDDING_KEY` / `EMBEDDING_BASE_URL`：嵌入模型 API（默认配置为硅基流动）

## 架构设计

### 核心流程

```
main.py → SessionManager → Session → Agent → ChatOpenAI
                    ↓           ↓
              (memory_mcp)   (fetch_mcp, file_mcp)
```

### 关键组件

| 组件 | 文件 | 职责 |
|-----------|------|----------------|
| SessionManager | `session.py` | 管理多会话，将会话索引持久化到 `memory/index.json` |
| Session | `session.py` | 每个会话拥有独立的 Memory MCP 和 Agent 实例 |
| Agent | `agent.py` | 协调 LLM 和 MCP 工具，处理工具调用循环 |
| ChatOpenAI | `chat_openai.py` | LLM 包装器，支持流式输出和工具调用解析 |
| MCPClient | `mcp_client.py` | MCP 客户端，使用后台任务管理连接生命周期 |
| FAISSRetriever | `embedding_faiss.py` | RAG 检索，基于 FAISS 持久化存储 |

### MCP 架构

`MCPClient` 使用后台 asyncio 任务来保持 stdio 连接，避免 anyio/CancelScope 的作用域问题。工具调用通过 `asyncio.Queue` 分派给后台任务执行。

### RAG 架构

使用 FAISS 进行持久化向量存储：

```
knowledge/
├── *.md, *.pdf, *.xlsx, *.csv    # 源文档
└── embedding/
    ├── index.json                # 文件元数据（修改时间、哈希值）
    ├── vectors.faiss             # FAISS 二进制索引
    └── metadata.json             # 文本内容映射
```

关键类：
- `FAISSRetriever`：高层接口，协调初始化流程
- `IndexManager`：检测文件变更（新增/修改/删除），支持增量更新
- `FAISSVectorStore`：FAISS 索引操作（保存/加载/搜索）
- `DocumentChunker`：递归文本切片，支持重叠

索引重建触发条件：文件修改、新增文件、删除文件或更换嵌入模型。

### 会话管理

会话通过 `memory/index.json` 持久化。每个会话：
- 拥有专用的 Memory MCP，存储隔离（`memory/sessions/{name}.jsonl`）
- 包含独立的 Agent 实例
- 支持延迟初始化（恢复的会话在首次使用时才连接）

### 交互命令

运行 `main.py` 时可使用以下命令：
- `/new [名称]` - 创建新会话
- `/list` - 列出所有会话
- `/switch <编号>` - 切换到指定会话
- `/delete <编号>` - 删除指定会话
- `/clear` - 清空当前会话的对话历史
- `/history` - 查看对话历史
- `/rag` - 切换 RAG 模式（使用 knowledge/ 目录，不使用会话记忆）
- `/help` - 显示帮助信息

## 重要实现细节

### 异步模式

整个代码库使用 `asyncio`。MCP 客户端连接通过后台任务管理，以避免 anyio 作用域违规。

### 工具调用流程

1. `Agent.invoke()` 发送提示词给 `ChatOpenAI.chat()`
2. 如果 LLM 返回 `toolCalls`，Agent 路由到对应的 MCP 客户端
3. 工具结果通过 `append_tool_result()` 追加到消息历史
4. 循环继续调用 `chat()`（不传提示词），直到没有更多工具调用

### 文档处理

`read_doc.py` 支持：`.md`、`.txt`、`.pdf`、`.xlsx`、`.xls`、`.csv`

切片使用递归字符分割，可配置分隔符优先级（段落 → 句子 → 单词）。

### 记忆规则（系统提示词）

默认系统提示词要求智能代理：
1. 对话开始时通过 `search_nodes`/`read_graph` 查询记忆
2. 对话结束前通过 `create_entities`/`add_observations`/`create_relations` 存储重要信息
3. 用户说"不用记"或为网页搜索结果时跳过存储
