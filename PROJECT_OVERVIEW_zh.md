# MMRAG 项目介绍

MMRAG 是一个 GitHub 多 Agent 知识库系统，核心目标是解决“开源仓库信息分散、文档与讨论上下文割裂、AI 回答难以验证”这类痛点。传统 RAG 往往只读取 README 或少量文档，无法覆盖 Issue 和 PR 中真正有价值的设计讨论、bug 成因和变更背景。这个项目会同时索引仓库 Markdown 文档、Issue 正文与评论、PR 描述与评审讨论，并统一写入向量库和元数据目录，使系统既能回答“官方文档怎么说”，也能回答“最近社区和维护者是怎么讨论这个问题的”。

核心逻辑流不是单次 prompt，而是完整的多 Agent 协作链路：Router Agent 先判断问题更适合从文档、Issue、PR 还是混合来源回答；Query Planner Agent 把原问题拆成 1 到 3 个检索子问题；Retriever Agents 并行检索不同来源的证据；Synthesis Agent 基于证据生成带引用的中文回答；Critic Agent 再对 groundedness 做自检，发现证据不足时会拒答或触发一次重试。整个流程会记录 trace_id、检索命中、步骤耗时、token usage 和最终 citations，便于演示、审计和迭代优化。

这个项目的工程价值在于它不绑定任何单一模型平台，只要求用户提供 OpenAI-compatible 的 LLM 和 Embedding 接口，因此可以灵活切换到不同服务或兼容网关。同时它具备 CLI、API、Docker Compose、评测脚本和项目说明文档，适合直接作为一个完整的 Agent 项目示例。
