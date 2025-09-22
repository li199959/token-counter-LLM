"""Serverless endpoint returning available models for Vercel deployments."""

from __future__ import annotations

from http import HTTPStatus
from http.server import BaseHTTPRequestHandler

from ._shared import get_service, send_empty, send_json


class handler(BaseHTTPRequestHandler):  # noqa: N801 - Vercel naming requirement
    def log_message(self, format, *args):  # pragma: no cover - silence logs in tests
        return

    def do_OPTIONS(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        send_empty(self)

    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        service = get_service()
        payload = {"models": service.list_models()}
        send_json(self, HTTPStatus.OK, payload)
