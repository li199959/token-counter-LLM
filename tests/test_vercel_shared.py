import io
import json

from http import HTTPStatus

from api import _shared


class DummyHandler:
    def __init__(self):
        self.status = None
        self.headers = []
        self.wfile = io.BytesIO()

    def send_response(self, status):
        self.status = status

    def send_header(self, key, value):
        self.headers.append((key, value))

    def end_headers(self):
        pass


def test_get_service_is_cached():
    service_a = _shared.get_service()
    service_b = _shared.get_service()
    assert service_a is service_b


def test_send_json_writes_payload():
    handler = DummyHandler()
    _shared.send_json(handler, HTTPStatus.OK, {"ok": True})
    body = json.loads(handler.wfile.getvalue().decode("utf-8"))
    assert handler.status == HTTPStatus.OK.value
    assert body == {"ok": True}
    assert ("Access-Control-Allow-Origin", "*") in handler.headers


def test_send_empty_returns_no_body():
    handler = DummyHandler()
    _shared.send_empty(handler)
    assert handler.status == HTTPStatus.NO_CONTENT.value
    assert handler.wfile.getvalue() == b""
