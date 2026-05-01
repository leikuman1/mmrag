from __future__ import annotations

import json
import re
import uuid
from typing import Any, TypedDict

from mmrag.model_providers.base import ChatMessage, ChatModelProvider
from mmrag.models import AnswerResponse, Citation, SearchHit
from mmrag.retrieval.service import RetrievalService, infer_source_types
from mmrag.storage.catalog import CatalogStore
from mmrag.utils import compact_text

from .tracing import TraceRecorder


class WorkflowState(TypedDict, total=False):
    trace_id: str
    session_id: str | None
    repo: str
    question: str
    route: str
    subqueries: list[str]
    hits: list[SearchHit]
    answer: str
    citations: list[Citation]
    follow_up_question: str | None
    grounded: bool
    confidence: float
    needs_retry: bool
    critique_reason: str
    retries: int
    trace_steps: list


def _dedupe_hits(hits: list[SearchHit], limit: int = 6) -> list[SearchHit]:
    seen: dict[str, SearchHit] = {}
    for hit in hits:
        current = seen.get(hit.chunk.id)
        if current is None or hit.score > current.score:
            seen[hit.chunk.id] = hit
    ranked = sorted(seen.values(), key=lambda item: item.score, reverse=True)
    return ranked[:limit]


class AgentWorkflow:
    def __init__(
        self,
        retrieval: RetrievalService,
        catalog: CatalogStore,
        chat_model: ChatModelProvider | None = None,
    ) -> None:
        self.retrieval = retrieval
        self.catalog = catalog
        self.chat_model = chat_model
        self.graph = self._build_graph()

    def _build_graph(self):
        try:
            from langgraph.graph import END, StateGraph
        except ImportError:
            return None

        graph = StateGraph(WorkflowState)
        graph.add_node("route", self._route_node)
        graph.add_node("plan", self._plan_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("synthesize", self._synthesize_node)
        graph.add_node("critic", self._critic_node)
        graph.set_entry_point("route")
        graph.add_edge("route", "plan")
        graph.add_edge("plan", "retrieve")
        graph.add_edge("retrieve", "synthesize")
        graph.add_edge("synthesize", "critic")
        graph.add_conditional_edges(
            "critic",
            lambda state: "retry" if state.get("needs_retry") and state.get("retries", 0) <= 1 else "finish",
            {"retry": "plan", "finish": END},
        )
        return graph.compile()

    def _route_node(self, state: WorkflowState) -> WorkflowState:
        recorder = TraceRecorder()
        with recorder.span("route", "Router Agent", state["question"]) as trace:
            route = self._route_question(state["question"])
            trace["output_summary"] = route
        state["route"] = route
        state["trace_steps"] = state.get("trace_steps", []) + recorder.steps
        return state

    def _plan_node(self, state: WorkflowState) -> WorkflowState:
        recorder = TraceRecorder()
        with recorder.span("plan", "Query Planner Agent", state["question"]) as trace:
            subqueries, token_usage = self._plan_queries(
                question=state["question"],
                route=state["route"],
                retries=state.get("retries", 0),
                critique_reason=state.get("critique_reason", ""),
            )
            trace["output_summary"] = json.dumps(subqueries, ensure_ascii=False)
            trace["token_usage"] = token_usage
        state["subqueries"] = subqueries
        state["trace_steps"] = state.get("trace_steps", []) + recorder.steps
        return state

    def _retrieve_node(self, state: WorkflowState) -> WorkflowState:
        recorder = TraceRecorder()
        with recorder.span("retrieve", "Retriever Agents", json.dumps(state["subqueries"], ensure_ascii=False)) as trace:
            hits: list[SearchHit] = []
            source_types = infer_source_types(state["question"])
            for subquery in state["subqueries"]:
                hits.extend(self.retrieval.retrieve(state["repo"], subquery, source_types=source_types))
            hits = _dedupe_hits(hits)
            trace["output_summary"] = f"{len(hits)} evidence chunks"
        state["hits"] = hits
        state["trace_steps"] = state.get("trace_steps", []) + recorder.steps
        return state

    def _synthesize_node(self, state: WorkflowState) -> WorkflowState:
        recorder = TraceRecorder()
        with recorder.span("synthesize", "Synthesis Agent", state["question"]) as trace:
            answer, citations, follow_up_question, confidence, token_usage = self._synthesize_answer(
                question=state["question"],
                hits=state.get("hits", []),
            )
            trace["output_summary"] = compact_text(answer, 140)
            trace["token_usage"] = token_usage
        state["answer"] = answer
        state["citations"] = citations
        state["follow_up_question"] = follow_up_question
        state["confidence"] = confidence
        state["trace_steps"] = state.get("trace_steps", []) + recorder.steps
        return state

    def _critic_node(self, state: WorkflowState) -> WorkflowState:
        recorder = TraceRecorder()
        with recorder.span("critic", "Critic Agent", state["answer"]) as trace:
            grounded, needs_retry, confidence, critique_reason, token_usage = self._critic_review(
                question=state["question"],
                answer=state["answer"],
                hits=state.get("hits", []),
                citations=state.get("citations", []),
            )
            trace["output_summary"] = critique_reason or ("grounded" if grounded else "not grounded")
            trace["token_usage"] = token_usage
        state["grounded"] = grounded
        state["needs_retry"] = needs_retry
        state["confidence"] = min(state.get("confidence", 0.0), confidence)
        state["critique_reason"] = critique_reason
        state["retries"] = state.get("retries", 0) + (1 if needs_retry else 0)
        state["trace_steps"] = state.get("trace_steps", []) + recorder.steps
        return state

    def _route_question(self, question: str) -> str:
        source_types = infer_source_types(question)
        if not source_types or len(source_types) > 1:
            return "mixed"
        return source_types[0]

    def _heuristic_plan(self, question: str, route: str, critique_reason: str = "") -> list[str]:
        queries = [question.strip()]
        if route == "doc":
            queries.append(f"documentation answer for: {question}")
        elif route == "issue":
            queries.append(f"issues and bugs related to: {question}")
        elif route == "pr":
            queries.append(f"pull request discussion related to: {question}")
        else:
            queries.append(f"repository documentation and discussions about: {question}")
        if critique_reason:
            queries.append(f"{question}. Extra focus: {critique_reason}")
        unique: list[str] = []
        for item in queries:
            normalized = item.strip()
            if normalized and normalized not in unique:
                unique.append(normalized)
        return unique[:3]

    def _plan_queries(self, question: str, route: str, retries: int, critique_reason: str) -> tuple[list[str], int]:
        if self.chat_model is None:
            return self._heuristic_plan(question, route, critique_reason), 0
        prompt = (
            "You are a query planner for a GitHub RAG agent. Return JSON with a single key "
            '"subqueries" whose value is a list of 1 to 3 search-ready subqueries. Keep them short and distinct.'
        )
        user_prompt = json.dumps(
            {
                "question": question,
                "route": route,
                "retries": retries,
                "critique_reason": critique_reason,
            },
            ensure_ascii=False,
        )
        result = self.chat_model.generate(
            [ChatMessage(role="system", content=prompt), ChatMessage(role="user", content=user_prompt)]
        )
        parsed = self._extract_json_object(result.text)
        subqueries = parsed.get("subqueries") if isinstance(parsed, dict) else None
        if not isinstance(subqueries, list) or not subqueries:
            return self._heuristic_plan(question, route, critique_reason), result.total_tokens
        cleaned = [str(item).strip() for item in subqueries if str(item).strip()]
        return cleaned[:3] or self._heuristic_plan(question, route, critique_reason), result.total_tokens

    def _synthesize_answer(
        self,
        question: str,
        hits: list[SearchHit],
    ) -> tuple[str, list[Citation], str | None, float, int]:
        citations = [
            Citation(
                source_type=hit.chunk.source_type,
                title=hit.chunk.title,
                url=hit.chunk.url,
                snippet=hit.chunk.snippet,
            )
            for hit in _dedupe_hits(hits, limit=4)
        ]
        if not hits:
            answer = "无法根据当前索引内容确认答案。请先扩大索引范围，或补充相关仓库文档、Issue、PR 讨论。"
            return answer, [], "是否需要我建议应补充哪些来源？", 0.15, 0
        evidence = "\n\n".join(
            [
                f"[{index}] {hit.chunk.title}\nURL: {hit.chunk.url}\nSnippet: {hit.chunk.snippet}\nText: {compact_text(hit.chunk.text, 900)}"
                for index, hit in enumerate(_dedupe_hits(hits), start=1)
            ]
        )
        if self.chat_model is None:
            answer = (
                "基于当前索引证据，可以确认以下信息：\n"
                + "\n".join(
                    f"- {hit.chunk.title}: {compact_text(hit.chunk.snippet, 120)}"
                    for hit in _dedupe_hits(hits, limit=3)
                )
            )
            confidence = min(0.85, 0.45 + 0.1 * len(citations))
            return answer, citations, None, confidence, 0
        prompt = (
            "You are the Synthesis Agent for a GitHub repository RAG system. "
            "Answer only from the provided evidence. Mention conflicts explicitly. "
            "If the evidence is incomplete, say so clearly instead of guessing."
        )
        result = self.chat_model.generate(
            [
                ChatMessage(role="system", content=prompt),
                ChatMessage(
                    role="user",
                    content=f"Question:\n{question}\n\nEvidence:\n{evidence}\n\nReturn a concise answer in Chinese.",
                ),
            ]
        )
        answer = result.text.strip() or "无法根据当前证据给出稳定答案。"
        confidence = min(0.95, 0.5 + 0.08 * len(citations))
        return answer, citations, None, confidence, result.total_tokens

    def _critic_review(
        self,
        question: str,
        answer: str,
        hits: list[SearchHit],
        citations: list[Citation],
    ) -> tuple[bool, bool, float, str, int]:
        if not hits:
            return False, False, 0.1, "No evidence was retrieved.", 0
        if self.chat_model is None:
            grounded = bool(citations) and "无法根据当前索引内容确认答案" not in answer
            needs_retry = not grounded
            reason = "Need more evidence or a stronger citation set." if needs_retry else "Answer is grounded in retrieved evidence."
            confidence = min(0.9, 0.4 + 0.1 * len(citations))
            return grounded, needs_retry, confidence, reason, 0
        prompt = (
            "You are the Critic Agent. Return JSON with keys grounded, needs_retry, confidence, critique_reason. "
            "Set grounded to true only if the answer is fully supported by evidence."
        )
        evidence = "\n".join(f"- {hit.chunk.title}: {hit.chunk.snippet}" for hit in _dedupe_hits(hits, limit=4))
        result = self.chat_model.generate(
            [
                ChatMessage(role="system", content=prompt),
                ChatMessage(
                    role="user",
                    content=f"Question: {question}\n\nAnswer: {answer}\n\nEvidence:\n{evidence}",
                ),
            ]
        )
        parsed = self._extract_json_object(result.text)
        if not parsed:
            grounded = bool(citations)
            return grounded, not grounded, 0.55, "Critic model returned invalid JSON.", result.total_tokens
        grounded = bool(parsed.get("grounded"))
        needs_retry = bool(parsed.get("needs_retry"))
        confidence = float(parsed.get("confidence", 0.5) or 0.5)
        critique_reason = str(parsed.get("critique_reason", "") or "")
        return grounded, needs_retry, confidence, critique_reason, result.total_tokens

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if not match:
                return {}
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return {}

    def _run_sequential(self, initial_state: WorkflowState) -> WorkflowState:
        state = dict(initial_state)
        state = self._route_node(state)
        state = self._plan_node(state)
        state = self._retrieve_node(state)
        state = self._synthesize_node(state)
        state = self._critic_node(state)
        if state.get("needs_retry") and state.get("retries", 0) <= 1:
            state = self._plan_node(state)
            state = self._retrieve_node(state)
            state = self._synthesize_node(state)
            state = self._critic_node(state)
        return state

    def answer(self, repo: str, question: str, session_id: str | None = None) -> AnswerResponse:
        initial_state: WorkflowState = {
            "trace_id": str(uuid.uuid4()),
            "session_id": session_id,
            "repo": repo,
            "question": question,
            "retries": 0,
            "trace_steps": [],
        }
        if self.graph is None:
            state = self._run_sequential(initial_state)
        else:
            state = self.graph.invoke(initial_state)
        response = AnswerResponse(
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            confidence=float(state.get("confidence", 0.0) or 0.0),
            trace_id=state["trace_id"],
            follow_up_question=state.get("follow_up_question"),
            grounded=bool(state.get("grounded", False)),
            needs_more_context=not bool(state.get("grounded", False)),
        )
        self.catalog.save_trace(state["trace_id"], session_id, repo, response, list(state.get("trace_steps", [])))
        return response
