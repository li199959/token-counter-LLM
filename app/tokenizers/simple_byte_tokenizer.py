"""Simple tokenizer that counts raw UTF-8 bytes."""

from __future__ import annotations

from typing import Sequence

from .base import TokenizerAdapter


class ByteTokenizer(TokenizerAdapter):
    """Tokenizer that treats each UTF-8 byte as an individual token."""

    def __init__(self, name: str = "byte-tokenizer") -> None:
        super().__init__(name=name)

    def tokenize(self, text: str) -> Sequence[int]:
        if not text:
            return []
        return list(text.encode("utf-8"))
