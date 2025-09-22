"""Tokenizer implementation using configurable regular expressions."""

from __future__ import annotations

import re
from typing import List, Sequence

from .base import TokenizerAdapter

_DEFAULT_PATTERN = (
    r"[\u4e00-\u9fff]"  # CJK Unified Ideographs
    r"|[\u3040-\u30ff]"  # Japanese Hiragana/Katakana
    r"|[\uac00-\ud7af]"  # Hangul syllables
    r"|[A-Za-z]+(?:'[A-Za-z]+)?"  # latin words with optional apostrophes
    r"|[0-9]+"  # numbers
    r"|_+"  # underscores sequences
    r"|[^\w\s]"  # punctuation and symbols
    r"|[\s]+"  # whitespace chunks
)


class RegexTokenizer(TokenizerAdapter):
    """A configurable tokenizer powered by regular expressions."""

    def __init__(
        self,
        name: str,
        pattern: str | None = None,
        normalize_lowercase: bool = False,
        keep_whitespace: bool = False,
        collapse_whitespace: bool = False,
    ) -> None:
        super().__init__(name=name)
        self._normalize_lowercase = normalize_lowercase
        self._keep_whitespace = keep_whitespace
        self._collapse_whitespace = collapse_whitespace
        self._pattern = re.compile(pattern or _DEFAULT_PATTERN, re.UNICODE)

    def tokenize(self, text: str) -> Sequence[str]:
        if not text:
            return []
        if self._normalize_lowercase:
            text = text.lower()

        raw_tokens: List[str] = []
        for match in self._pattern.finditer(text):
            token = match.group(0)
            if not self._keep_whitespace and token.isspace():
                continue
            if self._collapse_whitespace and token.isspace():
                if raw_tokens and raw_tokens[-1] == "<ws>":
                    continue
                raw_tokens.append("<ws>")
            else:
                raw_tokens.append(token)
        return raw_tokens
