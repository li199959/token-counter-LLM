"""Tokenizer adapter that loads Hugging Face repositories on demand."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from .base import TokenizerAdapter

_DEFAULT_USER_AGENT = "token-counter-llm/0.1"


class MissingDependencyError(RuntimeError):
    """Raised when optional Hugging Face dependencies are unavailable."""


class TokenizerDownloadError(RuntimeError):
    """Raised when the tokenizer assets cannot be downloaded."""


def _import_hf_tokenizer():
    """Import the optional :mod:`tokenizers` dependency."""

    try:
        from tokenizers import Tokenizer as HFTokenizer  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised in tests
        raise MissingDependencyError(
            "The 'tokenizers' package is required to use Hugging Face tokenizers."
            " Install it via 'pip install tokenizers'."
        ) from exc
    return HFTokenizer


@dataclass(frozen=True)
class _TokenizerLocation:
    """Resolved location of the cached tokenizer file."""

    path: Path
    from_cache: bool


class HuggingFaceTokenizer(TokenizerAdapter):
    """Tokenizer that mirrors Hugging Face repositories."""

    def __init__(
        self,
        name: str,
        *,
        repo_id: str,
        revision: str = "main",
        tokenizer_file: str = "tokenizer.json",
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        local_tokenizer_path: str | Path | None = None,
        add_special_tokens: bool = False,
        user_agent: str | None = None,
        download_timeout: float = 30.0,
        auth_token: str | None = None,
        auth_token_env: str | Sequence[str] | None = None,
    ) -> None:
        super().__init__(name=name)
        if not repo_id:
            raise ValueError("'repo_id' option must be provided for HuggingFaceTokenizer")

        self._repo_id = repo_id
        self._revision = revision or "main"
        self._tokenizer_file = tokenizer_file or "tokenizer.json"
        self._cache_root = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "token-counter-llm"
        self._local_files_only = bool(local_files_only)
        self._local_tokenizer_path = Path(local_tokenizer_path).expanduser() if local_tokenizer_path else None
        self._add_special_tokens = bool(add_special_tokens)
        self._user_agent = user_agent or _DEFAULT_USER_AGENT
        self._download_timeout = float(download_timeout)
        self._auth_token = self._resolve_auth_token(auth_token, auth_token_env)
        self._backend = None

    # ------------------------------------------------------------------
    # Helpers
    def _safe_repo_dir(self) -> Path:
        safe = self._repo_id.replace("/", "__")
        return self._cache_root / safe / self._revision

    def _ensure_local_tokenizer(self) -> _TokenizerLocation:
        if self._local_tokenizer_path:
            if not self._local_tokenizer_path.exists():
                raise FileNotFoundError(
                    f"Local tokenizer file not found: {self._local_tokenizer_path}"
                )
            return _TokenizerLocation(self._local_tokenizer_path, from_cache=False)

        target_dir = self._safe_repo_dir()
        target_path = target_dir / self._tokenizer_file
        if target_path.exists():
            return _TokenizerLocation(target_path, from_cache=True)

        if self._local_files_only:
            raise TokenizerDownloadError(
                "Local files only was requested but tokenizer file is missing."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        return _TokenizerLocation(self._download_tokenizer_file(target_path), from_cache=False)

    def _download_tokenizer_file(self, target_path: Path) -> Path:
        url = f"https://huggingface.co/{self._repo_id}/resolve/{self._revision}/{self._tokenizer_file}"
        headers = {"User-Agent": self._user_agent}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self._download_timeout) as response:
                data = response.read()
        except urllib.error.HTTPError as exc:  # pragma: no cover - network failure path
            auth_hint = ""
            if exc.code in {401, 403}:
                auth_hint = (
                    " Repository access was denied."
                    " Provide a Hugging Face access token via the 'auth_token' option"
                    " or the HUGGINGFACE_TOKEN / HUGGINGFACEHUB_API_TOKEN environment variables."
                )
            raise TokenizerDownloadError(
                f"Failed to download tokenizer from {url}: HTTP {exc.code} {exc.reason}.{auth_hint}"
            ) from exc
        except urllib.error.URLError as exc:  # pragma: no cover - network failure path
            raise TokenizerDownloadError(f"Failed to download tokenizer from {url}: {exc}") from exc

        try:
            json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):  # pragma: no cover - defensive
            raise TokenizerDownloadError(
                "Downloaded tokenizer file is not valid JSON."
            )

        target_path.write_bytes(data)
        return target_path

    def _create_backend(self, tokenizer_path: Path):
        hf_tokenizer_cls = _import_hf_tokenizer()
        return hf_tokenizer_cls.from_file(str(tokenizer_path))

    def _get_backend(self):
        if self._backend is None:
            location = self._ensure_local_tokenizer()
            self._backend = self._create_backend(location.path)
        return self._backend

    # ------------------------------------------------------------------
    # TokenizerAdapter API
    def tokenize(self, text: str) -> Sequence[str]:
        if not text:
            return []

        backend = self._get_backend()
        encoding = backend.encode(text, add_special_tokens=self._add_special_tokens)
        tokens = getattr(encoding, "tokens", None)
        if callable(tokens):  # pragma: no cover - compatibility guard
            tokens = tokens()
        if tokens is None:
            raise RuntimeError("The Hugging Face backend did not return token data.")
        return list(tokens)

    # ------------------------------------------------------------------
    # Authentication helpers
    def _resolve_auth_token(
        self,
        explicit_token: str | None,
        env_names: str | Sequence[str] | None,
    ) -> str | None:
        if explicit_token:
            return explicit_token

        candidates: Iterable[str]
        if env_names is None:
            candidates = ()
        elif isinstance(env_names, str):
            candidates = (env_names,)
        else:
            candidates = env_names

        checked: set[str] = set()
        for name in (*candidates, "HUGGINGFACE_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HF_TOKEN"):
            if not name or name in checked:
                continue
            checked.add(name)
            value = os.getenv(name)
            if value:
                return value
        return None