"""
Module 10: Tokenization - Core Functionality Tests
===================================================
"""

import numpy as np
import pytest
import sys
from pathlib import Path


from tinytorch.foundation.tokenization import CharTokenizer


class TestTokenizerBasics:
    """Test basic tokenization functionality."""

    def test_tokenizer_encode(self):
        """
        Verify tokenizer converts text to IDs.
        """

        # Build vocab from test text
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])

        text = "hello world"
        token_ids = tokenizer.encode(text)

        assert isinstance(
            token_ids, (list, np.ndarray)
        ), "encode() should return list or array of IDs"
        assert all(
            isinstance(id, (int, np.integer)) for id in token_ids
        ), "Token IDs should be integers"

    def test_tokenizer_decode(self):
        """
        Verify tokenizer converts IDs back to text.
        """

        # Build vocab from test text
        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])

        text = "hello world"
        token_ids = tokenizer.encode(text)
        decoded = tokenizer.decode(token_ids)

        assert "hello" in decoded.lower() and "world" in decoded.lower(), (
            f"decode(encode(text)) should recover the text.\n"
            f"  Original: '{text}'\n"
            f"  Recovered: '{decoded}'"
        )

    def test_vocabulary_size(self):
        """
        Verify tokenizer has a defined vocabulary.
        """

        tokenizer = CharTokenizer()
        tokenizer.build_vocab(["hello world"])

        vocab_size = tokenizer.vocab_size
        assert (
            isinstance(vocab_size, int) and vocab_size > 0
        ), "Tokenizer should have positive vocab_size"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
