"""Command line entry-point for the token counter application."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import load_registry
from .server import serve
from .services.token_service import TokenService
from .tokenizers.registry import TokenizerRegistry


def _create_service(registry_path: str | None = None) -> TokenService:
    models = load_registry(Path(registry_path) if registry_path else None)
    registry = TokenizerRegistry()
    return TokenService(models=models, registry=registry)


def _cmd_list_models(args) -> int:
    service = _create_service(args.registry)
    payload = {"models": service.list_models()}
    json.dump(payload, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_count(args) -> int:
    service = _create_service(args.registry)
    text = args.text
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8")
    result = service.calculate(model_id=args.model, text=text)
    json.dump(result, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


def _cmd_serve(args) -> int:
    host = args.host
    port = int(args.port)
    serve(host=host, port=port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LLM token counter utilities")
    parser.add_argument("--registry", help="Path to custom model registry JSON", default=None)

    subparsers = parser.add_subparsers(dest="command", required=True)

    sp_list = subparsers.add_parser("models", help="List supported models")
    sp_list.set_defaults(func=_cmd_list_models)

    sp_count = subparsers.add_parser("count", help="Count tokens for input text")
    sp_count.add_argument("--model", required=True, help="Model identifier")
    sp_count.add_argument("--text", help="Text to tokenize", default="")
    sp_count.add_argument("--file", help="Path to file with text content")
    sp_count.set_defaults(func=_cmd_count)

    sp_serve = subparsers.add_parser("serve", help="Start HTTP API server")
    sp_serve.add_argument("--host", default="127.0.0.1")
    sp_serve.add_argument("--port", default="8000")
    sp_serve.set_defaults(func=_cmd_serve)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
