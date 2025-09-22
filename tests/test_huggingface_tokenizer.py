import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from app.tokenizers.huggingface_tokenizer import (
    HuggingFaceTokenizer,
    MissingDependencyError,
    TokenizerDownloadError,
)


def test_tokenize_uses_local_file(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")

    tokenizer = HuggingFaceTokenizer(
        name="stub-hf",
        repo_id="example/model",
        local_tokenizer_path=tokenizer_path,
    )

    tokens = tokenizer.tokenize("Hello   world\n")
    assert tokens == ["Hello", "world"]


@pytest.mark.no_stub_hf
def test_missing_dependency_raises(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer_path.write_text("{}", encoding="utf-8")

    tokenizer = HuggingFaceTokenizer(
        name="hf",
        repo_id="example/model",
        local_tokenizer_path=tokenizer_path,
    )

    with pytest.raises(MissingDependencyError):
        tokenizer.tokenize("hello")


@pytest.mark.no_stub_hf
def test_local_files_only_requires_existing_cache(tmp_path):
    tokenizer = HuggingFaceTokenizer(
        name="hf-cache",
        repo_id="example/model",
        cache_dir=tmp_path,
        local_files_only=True,
    )

    with pytest.raises(TokenizerDownloadError):
        tokenizer.tokenize("hello")


@pytest.mark.no_stub_hf
def test_download_uses_auth_token_from_env(monkeypatch, tmp_path):
    class DummyEncoding:
        def __init__(self, tokens):
            self._tokens = tokens

        def tokens(self):
            return self._tokens

    class DummyBackend:
        def encode(self, text, add_special_tokens=False):
            parts = text.split()
            if add_special_tokens:
                parts = ["<bos>", *parts, "<eos>"]
            return DummyEncoding(parts)

    class DummyTokenizerModule:
        @staticmethod
        def from_file(path):
            assert Path(path).exists()
            return DummyBackend()

    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=DummyTokenizerModule))

    captured_headers = {}

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover - nothing to clean up
            return False

        def read(self):
            return b"{}"

    def fake_urlopen(request, timeout):
        captured_headers["Authorization"] = request.get_header("Authorization")
        captured_headers["User-Agent"] = request.get_header("User-agent")
        return DummyResponse()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("TEST_HF_TOKEN", "hf_dummy_token")

    tokenizer = HuggingFaceTokenizer(
        name="hf-auth",
        repo_id="example/model",
        cache_dir=tmp_path,
        auth_token_env="TEST_HF_TOKEN",
    )

    tokens = tokenizer.tokenize("你好 世界")
    assert tokens == ["你好", "世界"]
    assert captured_headers.get("Authorization") == "Bearer hf_dummy_token"
    assert captured_headers.get("User-Agent")


@pytest.mark.no_stub_hf
def test_download_error_forbidden_mentions_token(monkeypatch, tmp_path):
    class DummyTokenizerModule:
        @staticmethod
        def from_file(path):  # pragma: no cover - should not be reached
            raise AssertionError("from_file should not be called when download fails")

    monkeypatch.setitem(sys.modules, "tokenizers", types.SimpleNamespace(Tokenizer=DummyTokenizerModule))

    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", hdrs=None, fp=None)

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    tokenizer = HuggingFaceTokenizer(
        name="hf-forbidden",
        repo_id="example/model",
        cache_dir=tmp_path,
    )

    with pytest.raises(TokenizerDownloadError) as exc_info:
        tokenizer.tokenize("test")

    message = str(exc_info.value)
    assert "Hugging Face access token" in message