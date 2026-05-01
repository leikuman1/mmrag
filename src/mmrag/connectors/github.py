from __future__ import annotations

import base64
import json
from typing import Iterable
from urllib import parse, request

from mmrag.models import SourceDocument


class GitHubConnector:
    api_root = "https://api.github.com"

    def __init__(
        self,
        token: str | None = None,
        user_agent: str = "mmrag/0.1",
        max_markdown_files: int = 200,
        max_issues: int = 100,
        max_pull_requests: int = 50,
    ) -> None:
        self.token = token
        self.user_agent = user_agent
        self.max_markdown_files = max_markdown_files
        self.max_issues = max_issues
        self.max_pull_requests = max_pull_requests

    def _headers(self) -> dict[str, str]:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": self.user_agent,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _request_json(self, path: str, params: dict | None = None) -> dict | list:
        url = f"{self.api_root}{path}"
        if params:
            url += "?" + parse.urlencode(params)
        req = request.Request(url, headers=self._headers())
        with request.urlopen(req, timeout=30) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)

    def _paged_json(
        self,
        path: str,
        *,
        base_params: dict | None = None,
        per_page: int = 100,
        max_items: int = 100,
    ) -> list[dict]:
        page = 1
        results: list[dict] = []
        while len(results) < max_items:
            params = {"per_page": per_page, "page": page}
            if base_params:
                params.update(base_params)
            data = self._request_json(path, params)
            if not isinstance(data, list) or not data:
                break
            results.extend(data)
            if len(data) < per_page:
                break
            page += 1
        return results[:max_items]

    def _decode_content(self, repo: str, path: str) -> str:
        data = self._request_json(f"/repos/{repo}/contents/{parse.quote(path, safe='/')}")
        if not isinstance(data, dict):
            return ""
        encoded = data.get("content", "")
        if data.get("encoding") != "base64" or not encoded:
            return ""
        cleaned = encoded.replace("\n", "")
        return base64.b64decode(cleaned).decode("utf-8", errors="replace")

    def _fetch_repo_default_branch(self, repo: str) -> str:
        data = self._request_json(f"/repos/{repo}")
        return str(data.get("default_branch", "main"))

    def _fetch_tree(self, repo: str, branch: str) -> list[dict]:
        data = self._request_json(f"/repos/{repo}/git/trees/{branch}", {"recursive": 1})
        if not isinstance(data, dict):
            return []
        return list(data.get("tree") or [])

    def fetch_repository_documents(self, repo: str) -> list[SourceDocument]:
        branch = self._fetch_repo_default_branch(repo)
        entries = self._fetch_tree(repo, branch)
        markdown_entries = [
            entry
            for entry in entries
            if entry.get("type") == "blob" and str(entry.get("path", "")).lower().endswith((".md", ".mdx"))
        ][: self.max_markdown_files]
        documents: list[SourceDocument] = []
        for entry in markdown_entries:
            path = entry["path"]
            content = self._decode_content(repo, path)
            if not content.strip():
                continue
            url = f"https://github.com/{repo}/blob/{branch}/{path}"
            documents.append(
                SourceDocument(
                    id=f"{repo}:doc:{path}",
                    repo=repo,
                    source_type="doc",
                    title=path,
                    url=url,
                    text=content,
                    updated_at=str(entry.get("sha", branch)),
                    path=path,
                    metadata={"path": path, "branch": branch, "sha": entry.get("sha")},
                )
            )
        return documents

    def _build_issue_document(self, repo: str, issue: dict, comments: Iterable[dict]) -> SourceDocument:
        labels = [label["name"] for label in issue.get("labels") or [] if isinstance(label, dict)]
        comment_text = "\n\n".join(
            f"Comment by {comment.get('user', {}).get('login', 'unknown')}:\n{comment.get('body', '').strip()}"
            for comment in comments
            if comment.get("body")
        )
        body = issue.get("body") or ""
        text = f"{issue.get('title', '')}\n\n{body}".strip()
        if comment_text:
            text += f"\n\nDiscussion\n\n{comment_text}"
        return SourceDocument(
            id=f"{repo}:issue:{issue['number']}",
            repo=repo,
            source_type="issue",
            title=f"Issue #{issue['number']}: {issue.get('title', '')}",
            url=issue.get("html_url", ""),
            text=text,
            updated_at=issue.get("updated_at", ""),
            author=(issue.get("user") or {}).get("login"),
            labels=labels,
            number=issue.get("number"),
            metadata={"number": issue.get("number"), "labels": labels, "author": (issue.get("user") or {}).get("login")},
        )

    def fetch_issues(self, repo: str) -> list[SourceDocument]:
        items = self._paged_json(
            f"/repos/{repo}/issues",
            base_params={"state": "all"},
            max_items=self.max_issues,
        )
        documents: list[SourceDocument] = []
        for issue in items:
            if issue.get("pull_request"):
                continue
            comments = self._paged_json(f"/repos/{repo}/issues/{issue['number']}/comments", max_items=100)
            documents.append(self._build_issue_document(repo, issue, comments))
        return documents

    def _build_pull_request_document(
        self,
        repo: str,
        pr: dict,
        issue_comments: Iterable[dict],
        review_comments: Iterable[dict],
    ) -> SourceDocument:
        labels = [label["name"] for label in pr.get("labels") or [] if isinstance(label, dict)]
        discussion_sections = []
        for comment in issue_comments:
            if comment.get("body"):
                discussion_sections.append(
                    f"Issue comment by {comment.get('user', {}).get('login', 'unknown')}:\n{comment.get('body', '').strip()}"
                )
        for comment in review_comments:
            if comment.get("body"):
                discussion_sections.append(
                    f"Review comment by {comment.get('user', {}).get('login', 'unknown')} on {comment.get('path', 'unknown file')}:\n{comment.get('body', '').strip()}"
                )
        text = f"{pr.get('title', '')}\n\n{pr.get('body') or ''}".strip()
        if discussion_sections:
            text += "\n\nDiscussion\n\n" + "\n\n".join(discussion_sections)
        return SourceDocument(
            id=f"{repo}:pr:{pr['number']}",
            repo=repo,
            source_type="pr",
            title=f"PR #{pr['number']}: {pr.get('title', '')}",
            url=pr.get("html_url", ""),
            text=text,
            updated_at=pr.get("updated_at", ""),
            author=(pr.get("user") or {}).get("login"),
            labels=labels,
            number=pr.get("number"),
            metadata={"number": pr.get("number"), "labels": labels, "author": (pr.get("user") or {}).get("login")},
        )

    def fetch_pull_requests(self, repo: str) -> list[SourceDocument]:
        items = self._paged_json(
            f"/repos/{repo}/pulls",
            base_params={"state": "all"},
            max_items=self.max_pull_requests,
        )
        documents: list[SourceDocument] = []
        for pr in items:
            number = pr["number"]
            issue_comments = self._paged_json(f"/repos/{repo}/issues/{number}/comments", max_items=100)
            review_comments = self._paged_json(f"/repos/{repo}/pulls/{number}/comments", max_items=100)
            documents.append(self._build_pull_request_document(repo, pr, issue_comments, review_comments))
        return documents

    def fetch_sources(self, repo: str, include: set[str]) -> list[SourceDocument]:
        documents: list[SourceDocument] = []
        if "docs" in include:
            documents.extend(self.fetch_repository_documents(repo))
        if "issues" in include:
            documents.extend(self.fetch_issues(repo))
        if "prs" in include:
            documents.extend(self.fetch_pull_requests(repo))
        return documents
