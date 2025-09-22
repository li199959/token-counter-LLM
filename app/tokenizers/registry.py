"""Tokenizer registry and factory helpers."""

from __future__ import annotations

from typing import Dict

from ..models import ModelSpec, TokenizerSpec
from .base import TokenizerAdapter
from .regex_tokenizer import RegexTokenizer
from .huggingface_tokenizer import HuggingFaceTokenizer


class UnknownTokenizerError(ValueError):
    """Raised when an unknown tokenizer type is requested."""


class TokenizerRegistry:
    """Factory responsible for building tokenizers from specifications."""

    def __init__(self) -> None:
        self._cache: Dict[str, TokenizerAdapter] = {}

    def get_tokenizer(self, spec: TokenizerSpec, cache_key: str | None = None) -> TokenizerAdapter:
        """Return a tokenizer instance for *spec* (cached by *cache_key* if given)."""

        key = cache_key or f"{spec.type}:{hash(frozenset(spec.options.items()))}"
        if key in self._cache:
            return self._cache[key]

        tokenizer = self._create_tokenizer(spec)
        self._cache[key] = tokenizer
        return tokenizer

    def _create_tokenizer(self, spec: TokenizerSpec) -> TokenizerAdapter:
        type_name = spec.type.lower()
        options = dict(spec.options)
        name = options.pop("name", type_name)

        if type_name == "regex":
            return RegexTokenizer(name=name, **options)
        if type_name == "byte":
            from .simple_byte_tokenizer import ByteTokenizer

            return ByteTokenizer(name=name)
        if type_name in {"huggingface", "hf"}:
            return HuggingFaceTokenizer(name=name, **options)
        raise UnknownTokenizerError(f"Unknown tokenizer type: {spec.type}")

    def invalidate(self, cache_key: str | None = None) -> None:
        """Remove cached tokenizers."""

        if cache_key is None:
            self._cache.clear()
        else:
            self._cache.pop(cache_key, None)


def get_tokenizer_for_model(model: ModelSpec, registry: TokenizerRegistry) -> TokenizerAdapter:
    """Convenience helper returning the tokenizer for *model*."""

    return registry.get_tokenizer(model.tokenizer, cache_key=model.model_id)
