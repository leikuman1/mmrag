from __future__ import annotations

import json

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from mmrag.config import AppConfig, ConfigurationError
from mmrag.runtime import build_runtime

from .schemas import ChatRequest, EvalRequest, IngestionRequest


def create_app(config: AppConfig | None = None) -> FastAPI:
    runtime = build_runtime(config)
    app = FastAPI(title="MMRAG", version="0.1.0")

    @app.get("/healthz")
    def healthcheck() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/ingestions/github")
    def ingest_github(payload: IngestionRequest) -> dict:
        try:
            result = runtime.ingestion_service.ingest_github(payload.repo, set(payload.include))
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.to_dict()

    @app.post("/v1/chat")
    def chat(payload: ChatRequest) -> dict:
        try:
            response = runtime.workflow.answer(payload.repo, payload.question, payload.session_id)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return response.to_dict()

    @app.post("/v1/chat/stream")
    def chat_stream(payload: ChatRequest) -> StreamingResponse:
        try:
            response = runtime.workflow.answer(payload.repo, payload.question, payload.session_id)
        except ConfigurationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        def event_stream():
            yield f"data: {json.dumps({'type': 'trace', 'trace_id': response.trace_id}, ensure_ascii=False)}\n\n"
            for line in response.answer.splitlines():
                if line.strip():
                    yield f"data: {json.dumps({'type': 'chunk', 'content': line}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done', 'response': response.to_dict()}, ensure_ascii=False)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @app.get("/v1/sources/{repo:path}")
    def list_sources(repo: str) -> dict:
        return runtime.catalog.get_ingestion_state(repo)

    @app.post("/v1/evals/run")
    def run_eval(payload: EvalRequest) -> dict:
        result = runtime.eval_runner.run(payload.repo, payload.suite_name)
        return result.to_dict()

    return app

