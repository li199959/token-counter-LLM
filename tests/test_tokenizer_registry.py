import pytest

from app.models import TokenizerSpec
from app.tokenizers.registry import TokenizerRegistry, UnknownTokenizerError


def test_registry_builds_huggingface_tokenizer():
    spec = TokenizerSpec(
        type="huggingface",
        options={"name": "demo", "repo_id": "example/model"},
    )
    registry = TokenizerRegistry()
    tokenizer = registry.get_tokenizer(spec)
    assert tokenizer.name == "demo"


def test_registry_rejects_non_hf_tokenizer():
    spec = TokenizerSpec(type="regex", options={})
    registry = TokenizerRegistry()
    with pytest.raises(UnknownTokenizerError):
        registry.get_tokenizer(spec)
