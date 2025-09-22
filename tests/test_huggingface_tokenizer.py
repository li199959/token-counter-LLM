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
