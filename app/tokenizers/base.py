"""Base classes for tokenizer adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Sequence


class TokenizerAdapter(ABC):
    """Common API for model-specific tokenizers."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def tokenize(self, text: str) -> Sequence[object]:
        """Split *text* into a sequence of tokens."""

    def count_tokens(self, text: str) -> int:
        """Return the number of tokens emitted for *text*."""

        return len(self.tokenize(text))

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"{self.__class__.__name__}(name={self.name!r})"
