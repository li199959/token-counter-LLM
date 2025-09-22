import unittest

from app.tokenizers.regex_tokenizer import RegexTokenizer


class RegexTokenizerTests(unittest.TestCase):
    def test_basic_tokenization(self):
        tokenizer = RegexTokenizer(name="test", keep_whitespace=False)
        tokens = tokenizer.tokenize("Hello, 世界! 123\nNew line.")
        self.assertEqual(
            tokens,
            [
                "Hello",
                ",",
                "世",
                "界",
                "!",
                "123",
                "New",
                "line",
                ".",
            ],
        )

    def test_tokenizer_with_whitespace(self):
        tokenizer = RegexTokenizer(name="test", keep_whitespace=True, collapse_whitespace=True)
        tokens = tokenizer.tokenize("Hi  there\n")
        self.assertEqual(tokens, ["Hi", "<ws>", "there", "<ws>"])


if __name__ == "__main__":
    unittest.main()
