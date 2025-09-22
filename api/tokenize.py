"""Serverless tokenization endpoint for Vercel deployments."""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

from ._shared import ModelNotFoundError, get_service, send_empty, send_json
from app.tokenizers.huggingface_tokenizer import (
    MissingDependencyError,
    TokenizerDownloadError,
)


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel naming requirement
    def log_message(self, format, *args):  # pragma: no cover - silence logs in tests
        return

    def do_OPTIONS(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        send_empty(self)

    def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_body = self.rfile.read(content_length) if content_length else b""
        try:
            payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
        except json.JSONDecodeError:
            send_json(self, HTTPStatus.BAD_REQUEST, {"error": "invalid json"})
            return

        model_id = payload.get("model") or payload.get("model_id")
        text = payload.get("text", "")
        if not model_id:
            send_json(self, HTTPStatus.BAD_REQUEST, {"error": "'model' is required"})
            return

        service = get_service()
        try:
            result = service.calculate(model_id=model_id, text=text)
        except ModelNotFoundError:
            send_json(self, HTTPStatus.NOT_FOUND, {"error": f"unknown model '{model_id}'"})
            return
        except (MissingDependencyError, TokenizerDownloadError) as exc:
            send_json(self, HTTPStatus.SERVICE_UNAVAILABLE, {"error": str(exc)})
            return

        send_json(self, HTTPStatus.OK, result)

    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        send_json(self, HTTPStatus.METHOD_NOT_ALLOWED, {"error": "POST only"})
