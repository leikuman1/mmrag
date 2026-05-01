from __future__ import annotations

from dataclasses import dataclass

from mmrag.agents.workflow import AgentWorkflow
from mmrag.config import AppConfig
from mmrag.connectors.github import GitHubConnector
from mmrag.evals.runner import EvalRunner
from mmrag.indexing.service import GitHubIngestionService
from mmrag.model_providers.openai_compatible import (
    OpenAICompatibleChatProvider,
    OpenAICompatibleEmbeddingProvider,
)
from mmrag.retrieval.service import RetrievalService
from mmrag.storage.catalog import CatalogStore
from mmrag.storage.vector_store import QdrantRestVectorStore


@dataclass(slots=True)
class RuntimeContainer:
    config: AppConfig
    catalog: CatalogStore
    ingestion_service: GitHubIngestionService
    retrieval_service: RetrievalService
    workflow: AgentWorkflow
    eval_runner: EvalRunner


def build_runtime(config: AppConfig | None = None) -> RuntimeContainer:
    config = config or AppConfig.from_env()
    config.ensure_storage_paths()

    catalog = CatalogStore(config.sqlite_path)
    connector = GitHubConnector(token=config.github_token)
    vector_store = QdrantRestVectorStore(config.qdrant_url, config.qdrant_collection, config.request_timeout)
    chat_model = OpenAICompatibleChatProvider(config.llm, config.request_timeout) if config.llm else None
    embedder = (
        OpenAICompatibleEmbeddingProvider(config.embedding, config.request_timeout) if config.embedding else None
    )
    ingestion_service = GitHubIngestionService(connector, catalog, vector_store, embedder)
    retrieval_service = RetrievalService(catalog, vector_store, embedder)
    workflow = AgentWorkflow(retrieval_service, catalog, chat_model)
    eval_runner = EvalRunner(workflow)
    return RuntimeContainer(
        config=config,
        catalog=catalog,
        ingestion_service=ingestion_service,
        retrieval_service=retrieval_service,
        workflow=workflow,
        eval_runner=eval_runner,
    )

