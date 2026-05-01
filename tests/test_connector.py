import unittest

from mmrag.connectors.github import GitHubConnector


class ConnectorTests(unittest.TestCase):
    def test_issue_document_normalization(self) -> None:
        connector = GitHubConnector()
        issue = {
            "number": 42,
            "title": "Retry loop bug",
            "body": "The workflow loops forever.",
            "updated_at": "2026-05-01T00:00:00Z",
            "html_url": "https://github.com/owner/repo/issues/42",
            "user": {"login": "alice"},
            "labels": [{"name": "bug"}],
        }
        comments = [{"body": "Confirmed on Windows.", "user": {"login": "bob"}}]
        document = connector._build_issue_document("owner/repo", issue, comments)
        self.assertEqual(document.source_type, "issue")
        self.assertEqual(document.number, 42)
        self.assertIn("Confirmed on Windows", document.text)

    def test_pull_request_document_normalization(self) -> None:
        connector = GitHubConnector()
        pr = {
            "number": 7,
            "title": "Add critic agent",
            "body": "Introduces a groundedness check.",
            "updated_at": "2026-05-01T00:00:00Z",
            "html_url": "https://github.com/owner/repo/pull/7",
            "user": {"login": "alice"},
            "labels": [{"name": "enhancement"}],
        }
        issue_comments = [{"body": "Looks good.", "user": {"login": "reviewer"}}]
        review_comments = [
            {"body": "Please rename this field.", "user": {"login": "reviewer"}, "path": "workflow.py"}
        ]
        document = connector._build_pull_request_document("owner/repo", pr, issue_comments, review_comments)
        self.assertEqual(document.source_type, "pr")
        self.assertEqual(document.number, 7)
        self.assertIn("Please rename this field", document.text)


if __name__ == "__main__":
    unittest.main()
