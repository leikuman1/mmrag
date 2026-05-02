# 架构说明

## 目标

MMRAG 的目标不是做一个泛化聊天机器人，而是做一个可证明 Agent 能力的 GitHub 知识系统：

- 能 ingest 多源 GitHub 内容
- 能把问题拆成多步检索与综合
- 能在证据不足时拒答
- 能保存完整 trace 方便演示、排查和回溯分析

## 组件

### 1. Data Connector

`GitHubConnector` 采集三类来源：

- 仓库 Markdown 文档
- Issue 正文与评论
- PR 描述、issue comments、review comments

所有来源都会统一成 `SourceDocument`。

### 2. Indexing Pipeline

`GitHubIngestionService` 执行：

1. fetch
2. normalize
3. chunk
4. embed
5. upsert

元数据包括：

- `repo`
- `source_type`
- `title`
- `url`
- `path`
- `number`
- `author`
- `labels`

### 3. Storage

- `SQLite`：保存 source catalog、chunks、trace、ingestion state
- `Qdrant`：保存向量与检索 payload

### 4. Retrieval

`RetrievalService` 采用三层策略：

1. 向量检索
2. metadata filter
3. SQLite 关键词回退

如果用户在问题中带 `#123`，系统会自动抽取编号并过滤相关 Issue / PR。

### 5. Agent Workflow

工作流默认定义为：

```text
Router -> Planner -> Retrieve -> Synthesize -> Critic
```

如果环境中已安装 `langgraph`，则优先使用 `StateGraph` 编排；否则退化为顺序执行版本，但接口保持一致。

### 6. Observability

每次回答会记录：

- `trace_id`
- step 名称
- agent 名称
- 输入摘要
- 输出摘要
- latency
- token usage

## 回答策略

系统有三条硬约束：

1. 回答必须由检索证据支撑
2. 来源冲突时必须显式说明
3. 证据不足时必须拒答，而不是补完想象

## 适配模型

项目不绑定任何模型平台。只要用户能提供 OpenAI-compatible 的：

- `chat/completions`
- `embeddings`

即可接入。这样可以直接切换到 OpenAI、DeepSeek 或任何兼容网关。
