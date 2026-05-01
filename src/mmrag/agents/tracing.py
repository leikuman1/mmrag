from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass

from mmrag.models import AgentStepTrace


@dataclass(slots=True)
class TracedCall:
    step: str
    agent: str
    input_summary: str
    started_at: float


class TraceRecorder:
    def __init__(self) -> None:
        self.steps: list[AgentStepTrace] = []

    @contextmanager
    def span(self, step: str, agent: str, input_summary: str):
        call = TracedCall(step=step, agent=agent, input_summary=input_summary, started_at=time.perf_counter())
        holder: dict[str, str | int] = {"output_summary": "", "token_usage": 0}
        yield holder
        latency_ms = (time.perf_counter() - call.started_at) * 1000
        self.steps.append(
            AgentStepTrace(
                step=step,
                agent=agent,
                input_summary=call.input_summary,
                output_summary=str(holder.get("output_summary", "")),
                latency_ms=latency_ms,
                token_usage=int(holder.get("token_usage", 0) or 0),
            )
        )

