import unittest

from mmrag.evals.runner import EvalRunner
from mmrag.models import AnswerResponse, Citation


class FakeWorkflow:
    def answer(self, repo: str, question: str, session_id: str | None = None) -> AnswerResponse:
        _ = (repo, question, session_id)
        return AnswerResponse(
            answer="This answer mentions agent, workflow, and install behavior.",
            citations=[Citation(source_type="doc", title="README.md", url="https://example.com", snippet="agent workflow")],
            confidence=0.9,
            trace_id="trace-1",
            grounded=True,
        )


class EvalTests(unittest.TestCase):
    def test_eval_runner_returns_summary(self) -> None:
        runner = EvalRunner(FakeWorkflow())
        result = runner.run("owner/repo", "demo")
        self.assertEqual(result.total_cases, 6)
        self.assertGreater(result.citation_validity, 0)


if __name__ == "__main__":
    unittest.main()
