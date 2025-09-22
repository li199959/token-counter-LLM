import unittest

from app.config import load_registry
from app.services.token_service import TokenService
from app.tokenizers.registry import TokenizerRegistry


class TokenServiceTests(unittest.TestCase):
    def setUp(self):
        models = load_registry()
        registry = TokenizerRegistry()
        self.service = TokenService(models=models, registry=registry)

    def test_list_models_contains_expected_ids(self):
        model_ids = {entry["id"] for entry in self.service.list_models()}
        self.assertTrue({"openai-gpt2", "deepseek-chat", "qwen-2-7b"}.issubset(model_ids))

    def test_calculate_returns_usage_and_cost_fields(self):
        result = self.service.calculate("openai-gpt2", "Token counting test")
        self.assertGreater(result["token_count"], 0)
        self.assertEqual(result["max_context"], 1024)
        self.assertEqual(result["usage_ratio"], result["token_count"] / 1024)
        self.assertGreaterEqual(result["pricing"]["estimated_input_cost"], 0)

    def test_calculate_handles_huggingface_model(self):
        result = self.service.calculate("deepseek-chat", "Hello\n\nworld")
        self.assertEqual(result["tokens"], ["Hello", "world"])
        self.assertEqual(result["token_count"], 2)


if __name__ == "__main__":
    unittest.main()
