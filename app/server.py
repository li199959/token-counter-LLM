"""Minimal HTTP server exposing the token counting service."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Callable

from .config import load_registry
from .services.token_service import ModelNotFoundError, TokenService
from .tokenizers.registry import TokenizerRegistry


def _load_frontend_html() -> str:
    """Load the bundled single-page frontend."""

    frontend_path = Path(__file__).resolve().parents[1] / "frontend" / "index.html"
    try:
        return frontend_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return """<!DOCTYPE html><html><body><h1>Token Counter</h1><p>frontend/index.html is missing.</p></body></html>"""


INDEX_HTML = _load_frontend_html()

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


def _build_handler(service: TokenService) -> Callable[..., BaseHTTPRequestHandler]:
    class TokenCounterHandler(BaseHTTPRequestHandler):
        def _write_common_headers(self) -> None:
            for header, value in _CORS_HEADERS.items():
                self.send_header(header, value)

        def _send_json(self, status: HTTPStatus, payload) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status.value)
            self._write_common_headers()
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _send_html(self, status: HTTPStatus, body: str) -> None:
            payload = body.encode("utf-8")
            self.send_response(status.value)
            self._write_common_headers()
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format, *args):  # pragma: no cover - quieter tests
            return

        def do_OPTIONS(self):  # noqa: N802 - required by BaseHTTPRequestHandler
            self.send_response(HTTPStatus.NO_CONTENT.value)
            self._write_common_headers()
            self.send_header("Content-Length", "0")
            self.end_headers()

        def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
            path = self.path.split("?", 1)[0]
            if path in {"", "/", "/index.html"}:
                self._send_html(HTTPStatus.OK, INDEX_HTML)
            elif path.rstrip("/") == "/models":
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
