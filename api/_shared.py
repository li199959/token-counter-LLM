"""Helpers shared by Vercel serverless handlers."""

from __future__ import annotations

import json
from functools import lru_cache
from http import HTTPStatus
from typing import Any, Dict

from app.config import load_registry
from app.services.token_service import ModelNotFoundError, TokenService
from app.tokenizers.registry import TokenizerRegistry


_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


@lru_cache()
def get_service() -> TokenService:
    """Return a cached :class:`TokenService` instance for serverless handlers."""

    models = load_registry()
    registry = TokenizerRegistry()
    return TokenService(models=models, registry=registry)


def send_json(handler, status: HTTPStatus, payload: Dict[str, Any]) -> None:
    """Serialize *payload* and write a JSON response with CORS headers."""

    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status.value)
    for header, value in _CORS_HEADERS.items():
        handler.send_header(header, value)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def send_empty(handler, status: HTTPStatus = HTTPStatus.NO_CONTENT) -> None:
    """Send an empty response body with CORS headers."""

    handler.send_response(status.value)
    for header, value in _CORS_HEADERS.items():
        handler.send_header(header, value)
    handler.send_header("Content-Length", "0")
    handler.end_headers()


__all__ = ["ModelNotFoundError", "get_service", "send_json", "send_empty"]
