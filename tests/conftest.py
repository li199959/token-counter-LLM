import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _DummyEncoding:
    def __init__(self, tokens):
        self.tokens = tokens


class _DummyBackend:
    def encode(self, text, add_special_tokens=False):
        normalized = " ".join(text.replace("\n", " ").split())
        core = normalized.split(" ") if normalized else []
        tokens = list(core)
        if add_special_tokens:
            tokens = ["<bos>", *tokens, "<eos>"]
        return _DummyEncoding(tokens)


@pytest.fixture(autouse=True)
def _stub_hf_tokenizer(monkeypatch, request):
    """Provide a lightweight stub when optional deps are missing."""

    if request.node.get_closest_marker("no_stub_hf"):
        return

    if importlib.util.find_spec("tokenizers") is not None:
        return

    from app.tokenizers import huggingface_tokenizer as hf_module

    def fake_create_backend(self, path):
        return _DummyBackend()

    def fake_download(self, target_path):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text("{}", encoding="utf-8")
        return target_path

    monkeypatch.setattr(hf_module.HuggingFaceTokenizer, "_create_backend", fake_create_backend)
    monkeypatch.setattr(hf_module.HuggingFaceTokenizer, "_download_tokenizer_file", fake_download)
