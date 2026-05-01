# Demo Script

这个脚本适合录一个 3-5 分钟的 Agent 项目展示视频。

## 1. 展示仓库结构

打开：

- `README.md`
- `docs/architecture.md`
- `src/mmrag/agents/workflow.py`
- `src/mmrag/connectors/github.py`

强调三点：

- 多 Agent，而不是单轮 prompt 包装
- GitHub 三类知识源统一索引
- 每次回答带 trace 和 citation

## 2. 展示环境变量

打开 `.env.example`，说明：

- 项目不内置模型
- 用户自行填 `LLM_*` 和 `EMBEDDING_*`
- 可以接任何 OpenAI-compatible 网关

这很适合说明项目工程化和平台无绑定能力。

## 3. 展示索引

运行：

```bash
mmrag ingest github --repo langchain-ai/langgraph --include docs,issues,prs
```

讲解：

- 文档、Issue、PR 一次性进入统一知识库
- chunk 会带元数据
- 向量进入 Qdrant，目录信息进入 SQLite

## 4. 展示问答

运行：

```bash
mmrag ask --repo langchain-ai/langgraph --question "README 里如何解释 multi-agent orchestration？"
```

再运行：

```bash
mmrag ask --repo langchain-ai/langgraph --question "最近有哪些 issue 在讨论错误处理？"
```

讲解：

- 不同问题由 Router Agent 分流
- Query Planner 会生成多个检索子问题
- 回答结果带 citation

## 5. 展示 trace

运行：

```bash
mmrag ask --repo langchain-ai/langgraph --question "最近的 PR 在改什么？" --save-trace data/trace.json
```

打开 trace 文件，展示：

- route
- plan
- retrieve
- synthesize
- critic

这一步最适合证明 Agent 是“真的在工作”，不是单条 prompt。

## 6. 展示评测

运行：

```bash
mmrag eval run --repo langchain-ai/langgraph --suite demo
```

强调：

- 有 citation validity
- 有 groundedness pass rate
- 有 refusal 策略

## 7. 展示申请回答

打开 `APPLICATION_ANSWER_zh.md`，说明这个项目已包含可直接提交的项目描述素材。

