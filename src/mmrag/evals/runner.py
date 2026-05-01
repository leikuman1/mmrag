from __future__ import annotations

import json
from importlib import resources

from mmrag.agents.workflow import AgentWorkflow
from mmrag.models import EvalCase, EvalCaseResult, EvalResult


class EvalRunner:
    def __init__(self, workflow: AgentWorkflow) -> None:
        self.workflow = workflow

    def _load_suite(self, suite_name: str) -> list[EvalCase]:
        with resources.files("mmrag.evals").joinpath(f"{suite_name}_suite.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return [
            EvalCase(
                case_id=item["case_id"],
                question=item["question"],
                expected_keywords=item.get("expected_keywords", []),
                expected_sources=item.get("expected_sources", []),
            )
            for item in payload
        ]

    def run(self, repo: str, suite_name: str = "demo") -> EvalResult:
        cases = self._load_suite(suite_name)
        results: list[EvalCaseResult] = []
        citation_pass = 0
        helpful_pass = 0
        grounded_pass = 0
        for case in cases:
            answer = self.workflow.answer(repo, case.question)
            citation_valid = bool(answer.citations)
            helpful = any(keyword.lower() in answer.answer.lower() for keyword in case.expected_keywords) if case.expected_keywords else bool(answer.answer.strip())
            grounded = answer.grounded or (citation_valid and "无法根据当前索引内容确认答案" not in answer.answer)
            passed = citation_valid and grounded and helpful
            if citation_valid:
                citation_pass += 1
            if helpful:
                helpful_pass += 1
            if grounded:
                grounded_pass += 1
            notes = "ok" if passed else "Missing expected keyword, citation, or groundedness signal."
            results.append(
                EvalCaseResult(
                    case_id=case.case_id,
                    question=case.question,
                    passed=passed,
                    citation_valid=citation_valid,
                    helpful=helpful,
                    grounded=grounded,
                    notes=notes,
                )
            )
        total = max(len(cases), 1)
        return EvalResult(
            suite_name=suite_name,
            total_cases=len(cases),
            helpfulness=helpful_pass / total,
            citation_validity=citation_pass / total,
            grounded_pass_rate=grounded_pass / total,
            cases=results,
        )

