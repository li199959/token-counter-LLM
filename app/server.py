"""Minimal HTTP server exposing the token counting service."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Callable

from .config import load_registry
from .services.token_service import ModelNotFoundError, TokenService
from .tokenizers.registry import TokenizerRegistry


def _build_handler(service: TokenService) -> Callable[..., BaseHTTPRequestHandler]:
    class TokenCounterHandler(BaseHTTPRequestHandler):
        def _send_json(self, status: HTTPStatus, payload):
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):  # pragma: no cover - quieter tests
            return

        def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path.rstrip("/") == "/models":
                self._send_json(HTTPStatus.OK, {"models": service.list_models()})
            else:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})

        def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
            if self.path.rstrip("/") != "/tokenize":
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "unknown endpoint"})
                return

            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length) if content_length else b""
            try:
                payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except json.JSONDecodeError:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
                return

            model_id = payload.get("model") or payload.get("model_id")
            text = payload.get("text", "")
            if not model_id:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "'model' is required"})
                return

            try:
                result = service.calculate(model_id=model_id, text=text)
            except ModelNotFoundError:
                self._send_json(HTTPStatus.NOT_FOUND, {"error": f"unknown model '{model_id}'"})
                return

            self._send_json(HTTPStatus.OK, result)

    return TokenCounterHandler


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start a blocking HTTP server."""

    registry = TokenizerRegistry()
    models = load_registry()
    service = TokenService(models=models, registry=registry)
    handler = _build_handler(service)
    with HTTPServer((host, port), handler) as httpd:
        httpd.serve_forever()
