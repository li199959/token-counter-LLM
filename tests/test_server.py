import json
import threading
import time
import unittest
from http.client import HTTPConnection
from http.server import HTTPServer

from app.config import load_registry
from app.server import _build_handler
from app.services.token_service import TokenService
from app.tokenizers.registry import TokenizerRegistry


class ServerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        models = load_registry()
        registry = TokenizerRegistry()
        service = TokenService(models=models, registry=registry)
        cls.httpd = HTTPServer(("127.0.0.1", 0), _build_handler(service))
        cls.port = cls.httpd.server_address[1]
        cls._thread = threading.Thread(target=cls.httpd.serve_forever, daemon=True)
        cls._thread.start()
        # Give the server a brief moment to ensure it is ready to accept requests
        time.sleep(0.05)

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls._thread.join()

    def _request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        conn = HTTPConnection("127.0.0.1", type(self).port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            payload = response.read()
            return (
                response.status,
                response.getheader("Content-Type"),
                payload,
                response.getheader("Access-Control-Allow-Origin"),
            )
        finally:
            conn.close()

    def test_root_route_serves_frontend(self):
        status, content_type, payload, cors = self._request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("text/html", content_type)
        self.assertEqual(cors, "*")
        body = payload.decode("utf-8")
        self.assertIn("<!DOCTYPE html>", body)
        self.assertIn("大模型 Token 计算器", body)

    def test_tokenize_endpoint_handles_request(self):
        payload = json.dumps({"model": "openai-gpt2", "text": "Hello world"}).encode("utf-8")
        status, content_type, body, cors = self._request(
            "POST",
            "/tokenize",
            body=payload,
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        self.assertIn("application/json", content_type)
        self.assertEqual(cors, "*")
        data = json.loads(body.decode("utf-8"))
        self.assertEqual(data["model"]["id"], "openai-gpt2")
        self.assertGreater(data["token_count"], 0)

    def test_options_request_returns_cors_headers(self):
        status, _, _, cors = self._request("OPTIONS", "/tokenize")
        self.assertEqual(status, 204)
        self.assertEqual(cors, "*")


if __name__ == "__main__":
    unittest.main()
