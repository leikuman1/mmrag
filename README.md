# MMRAG

MMRAG 是一个面向 GitHub 仓库文档、Issue、PR Discussion 的多 Agent 知识库 RAG 项目，重点展示：

- 长链路推理：问题路由、查询规划、检索、综合、批判式自检组成完整闭环
- 多 Agent 协作：`Router Agent -> Query Planner Agent -> Retriever Agents -> Synthesis Agent -> Critic Agent`
- 可验证输出：每次回答都返回 `citation`、`trace_id`、可追踪的 Agent 步骤
- 平台无绑定：项目只要求用户提供 OpenAI-compatible 的 `LLM` 与 `Embedding` 接口，不内置任何特定厂商逻辑

## 主要能力

- 索引 GitHub 仓库 Markdown 文档、Issue、PR 描述与评论
- 基于 `Qdrant + SQLite` 做向量检索、元数据过滤、关键词回退
- 提供 `CLI + FastAPI API + Docker Compose`
- 内置 demo 评测、项目说明文档、演示脚本和问题集

## 仓库结构

```text
src/mmrag/
  agents/            # Agent workflow, tracing
  api/               # FastAPI application and schemas
  cli/               # CLI entrypoint
  connectors/        # GitHub connector
  evals/             # Demo evaluation suites
  indexing/          # Chunking and ingestion flow
  model_providers/   # OpenAI-compatible chat / embedding adapters
  retrieval/         # Vector + keyword retrieval
  storage/           # SQLite catalog and Qdrant REST store
docs/
examples/
tests/
```

## 环境变量

项目本身不内置固定模型。用户自行提供兼容 OpenAI 接口的地址、密钥和模型名即可。

复制示例环境变量：

```bash
cp .env.example .env
```

最小必填项：

```env
LLM_BASE_URL=
LLM_API_KEY=
LLM_MODEL=

EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=

QDRANT_URL=http://localhost:6333
SQLITE_PATH=data/mmrag.sqlite3
```

可选：

```env
GITHUB_TOKEN=
MMRAG_DEFAULT_DEMO_REPO=langchain-ai/langgraph
```

## 安装

### 本地安装

```bash
python -m pip install -U pip
python -m pip install -e .
```

### Docker Compose

```bash
docker compose up --build
```

这会启动：

- `qdrant` 向量库
- `app` FastAPI 服务

## CLI 用法

索引 GitHub 仓库：

```bash
mmrag ingest github --repo langchain-ai/langgraph --include docs,issues,prs
```

提问：

```bash
mmrag ask --repo langchain-ai/langgraph --question "这个仓库怎么描述 multi-agent workflow？"
```

保存 trace：

```bash
mmrag ask --repo langchain-ai/langgraph --question "最近的 PR 在改什么？" --save-trace data/trace.json
```

运行 demo 评测：

```bash
mmrag eval run --repo langchain-ai/langgraph --suite demo
```

启动 API：

```bash
mmrag serve --host 0.0.0.0 --port 8000
```

## API

- `POST /v1/ingestions/github`
- `POST /v1/chat`
- `POST /v1/chat/stream`
- `GET /v1/sources/{repo}`
- `POST /v1/evals/run`

示例请求：

```json
{
  "repo": "langchain-ai/langgraph",
  "question": "README 里怎么解释 graph-based agent orchestration？"
}
```

## Agent 流程

1. `Router Agent` 判断问题更偏文档、Issue、PR 还是混合问题
2. `Query Planner Agent` 生成 1-3 个搜索子问题
3. `Retriever Agents` 并行检索 docs / issues / prs 相关证据
4. `Synthesis Agent` 基于证据生成带引用的中文回答
5. `Critic Agent` 检查 groundedness；若证据明显不足，最多重试一次

回答结果会带：

- `citations[]`
- `confidence`
- `trace_id`
- `follow_up_question`

## 验证

标准库自测：

```bash
python -m unittest discover -s tests -v
```

语法检查：

```bash
python -m compileall src
```

## 适合展示的亮点

- 多 Agent 协作，而不是单轮问答包装
- GitHub 多源知识整合，而不是只读 README
- 有 trace、评测和 refusal 机制，能证明不是“会说但不可验证”
- 对模型供应商零绑定，方便切换到任何兼容网关

更多细节见：

- [架构说明](docs/architecture.md)
- [演示脚本](docs/demo-script.md)
- [评测说明](docs/eval-report.md)
- [项目介绍](PROJECT_OVERVIEW_zh.md)
