from __future__ import annotations

import argparse
import json

from mmrag.api.app import create_app
from mmrag.config import AppConfig, ConfigurationError
from mmrag.runtime import build_runtime


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mmrag", description="MMRAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="Ingest repository content.")
    ingest_subparsers = ingest.add_subparsers(dest="ingest_command", required=True)
    github = ingest_subparsers.add_parser("github", help="Ingest GitHub repo data.")
    github.add_argument("--repo", required=True)
    github.add_argument("--include", default="docs,issues,prs")

    ask = subparsers.add_parser("ask", help="Ask a question against indexed repository content.")
    ask.add_argument("--repo", required=True)
    ask.add_argument("--question", required=True)
    ask.add_argument("--session-id")
    ask.add_argument("--save-trace")

    evaluate = subparsers.add_parser("eval", help="Run evaluation suites.")
    eval_subparsers = evaluate.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_subparsers.add_parser("run", help="Run an evaluation suite.")
    eval_run.add_argument("--repo")
    eval_run.add_argument("--suite", default="demo")

    serve = subparsers.add_parser("serve", help="Run the FastAPI service.")
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    return parser


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    config = AppConfig.from_env()
    runtime = build_runtime(config)

    try:
        if args.command == "ingest" and args.ingest_command == "github":
            include = {item.strip() for item in args.include.split(",") if item.strip()}
            result = runtime.ingestion_service.ingest_github(args.repo, include)
            _print_json(result.to_dict())
            return 0

        if args.command == "ask":
            response = runtime.workflow.answer(args.repo, args.question, args.session_id)
            _print_json(response.to_dict())
            if args.save_trace:
                trace = runtime.catalog.get_trace(response.trace_id)
                with open(args.save_trace, "w", encoding="utf-8") as handle:
                    json.dump(trace, handle, ensure_ascii=False, indent=2)
            return 0

        if args.command == "eval" and args.eval_command == "run":
            repo = args.repo or config.default_demo_repo
            result = runtime.eval_runner.run(repo, args.suite)
            _print_json(result.to_dict())
            return 0

        if args.command == "serve":
            import uvicorn

            host = args.host or config.host
            port = args.port or config.port
            uvicorn.run(create_app(config), host=host, port=port)
            return 0
    except ConfigurationError as exc:
        parser.exit(status=2, message=f"Configuration error: {exc}\n")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
