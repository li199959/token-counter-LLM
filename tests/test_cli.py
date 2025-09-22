import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import app.__main__ as cli


class CliTests(unittest.TestCase):
    def test_cli_count(self):
        with io.StringIO() as buffer:
            with redirect_stdout(buffer):
                exit_code = cli.main([
                    "--registry",
                    str(Path(__file__).resolve().parents[1] / "app" / "resources" / "model_registry.json"),
                    "count",
                    "--model",
                    "gpt-4o-mini",
                    "--text",
                    "Hello world",
                ])
            self.assertEqual(exit_code, 0)
            payload = json.loads(buffer.getvalue())
            self.assertEqual(payload["model"]["id"], "gpt-4o-mini")

    def test_cli_models_lists_entries(self):
        with io.StringIO() as buffer:
            with redirect_stdout(buffer):
                cli.main(["models"])
            payload = json.loads(buffer.getvalue())
            self.assertIn("models", payload)
            self.assertTrue(any(model["id"] == "qwen-2-7b" for model in payload["models"]))


if __name__ == "__main__":
    unittest.main()
