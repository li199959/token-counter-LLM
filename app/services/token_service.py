"""Token counting service that orchestrates tokenizers and models."""

from __future__ import annotations

from typing import Dict, Iterable, List

from ..models import ModelSpec
from ..tokenizers.registry import TokenizerRegistry, get_tokenizer_for_model


class ModelNotFoundError(KeyError):
    """Raised when a requested model is not registered."""


class TokenService:
    """High level API used by both CLI and HTTP interfaces."""

    def __init__(self, models: Iterable[ModelSpec], registry: TokenizerRegistry | None = None) -> None:
        self._models: Dict[str, ModelSpec] = {model.model_id: model for model in models}
        self._registry = registry or TokenizerRegistry()

    def list_models(self) -> List[Dict[str, object]]:
        return [model.to_dict() for model in self._models.values()]

    def get_model(self, model_id: str) -> ModelSpec:
        try:
            return self._models[model_id]
        except KeyError as exc:  # pragma: no cover - defensive
            raise ModelNotFoundError(model_id) from exc

    def calculate(self, model_id: str, text: str) -> Dict[str, object]:
        model = self.get_model(model_id)
        tokenizer = get_tokenizer_for_model(model, self._registry)
        tokens = tokenizer.tokenize(text)
        token_count = len(tokens)
        max_context = model.max_context
        usage_ratio = token_count / max_context if max_context else None
        overflow = max(token_count - max_context, 0) if max_context else 0

        pricing_info = None
        if model.pricing:
            pricing_info = model.pricing.to_dict()
            input_price = model.pricing.input_per_1k
            if input_price:
                pricing_info["estimated_input_cost"] = round((token_count / 1000) * input_price, 6)

        return {
            "model": model.to_dict(),
            "token_count": token_count,
            "tokens": tokens,
            "max_context": max_context,
            "usage_ratio": usage_ratio,
            "overflow": overflow,
            "pricing": pricing_info,
        }
