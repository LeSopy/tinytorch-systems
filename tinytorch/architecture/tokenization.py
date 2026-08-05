__all__ = [
    "KB_TO_BYTES",
    "Tokenizer",
    "CharTokenizer",
    "BPETokenizer",
    "create_tokenizer",
    "tokenize_dataset",
    "analyze_tokenization",
]

from collections import Counter
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

KB_TO_BYTES = 1024  # Kilobytes to bytes conversion


class Tokenizer:
    """Base tokenizer class providing the interface for all tokenizers."""

    TOK_UNKNOWN = "<UNK>"  # UNKNOWN
    TOK_EOW = "</w>"  # END OF WORD

    def encode(self, text: str) -> List[int]:
        """
        Convert text to a list of token IDs.
        """

        raise NotImplementedError(
            f"encode() not implemented in base Tokenizer class \n"
            f"Called encode() on abstract base class {self.__class__.__name__}\n"
            f"Tokenizer is an interface. Call with implementation like CharTokenizer or BPETokenizer\n"
        )

    def decode(self, token: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """

        raise NotImplementedError(
            f"decode() not implemented in base Tokenizer class \n"
            f"Called decode() on abstract base class {self.__class__.__name__}\n"
            f"Tokenizer is an interface. Call with implementation like CharTokenizer or BPETokenizer\n"
        )


class CharTokenizer(Tokenizer):
    """
    Character-level tokenizer that treats each character as a separate token.
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initialize character tokenizer.
        """

        if vocab is None:
            vocab = []

        # Add special unknown token
        self.vocab = [Tokenizer.TOK_UNKNOWN] + vocab
        self.vocab_size = len(self.vocab)

        # Create bidirectional mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

        # Store unknown token ID
        self.unk_id = 0

    def build_vocab(self, corpus: List[str]) -> None:
        """
        Build vocabulary from a corpus of text.
        """

        # Collect all unique characters
        all_chars = set()
        for text in corpus:
            all_chars.update(text)

        # Sort for consistent ordering
        unique_chars = sorted(all_chars)

        # Rebuild vocabulary with <UNK> token first
        self.vocab = [Tokenizer.TOK_UNKNOWN] + unique_chars
        self.vocab_size = len(self.vocab)

        # Rebuild mappings
        self.char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        self.id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    def encode(self, text: str) -> List[int]:
        """
        Encode text to list of character IDs.
        """

        tokens = []
        for char in text:
            tokens.append(self.char_to_id.get(char, self.unk_id))
        return tokens

    def decode(self, tokens: List[int]) -> str:
        """
        Decode list of token IDs back to text.
        """

        chars = []
        for token_id in tokens:
            # Use unknown token for invalid IDs
            char = self.id_to_char.get(token_id, Tokenizer.TOK_UNKNOWN)
            chars.append(char)
        return "".join(chars)


def _count_byte_pairs(word_tokens: Dict[str, List[str]], word_freq: Counter) -> Counter:
    """
    Count frequency of all adjacent token pairs across all words.
    """

    pair_counts = Counter()

    for word, freq in word_freq.items():
        tokens = word_tokens[word]
        # Count adjacent pairs
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            pair_counts[pair] += freq

    return pair_counts


def _merge_pair(word_tokens: Dict[str, List[str]], pair: Tuple[str, str]) -> str:
    """
    Merge the most frequent pair in all word token lists.
    """

    merged_token = pair[0] + pair[1]

    for word in word_tokens:
        tokens = word_tokens[word]
        new_tokens = []
        i = 0
        while i < len(tokens):
            if (
                i < len(tokens) - 1
                and tokens[i] == pair[0]
                and tokens[i + 1] == pair[1]
            ):
                # Merge pair
                new_tokens.append(merged_token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        word_tokens[word] = new_tokens

    return merged_token


class BPETokenizer(Tokenizer):
    """
    Byte Pair Encoding (BPE) tokenizer that learns subword units.
    """

    def __init__(self, vocab_size: int = 1000):
        """
        Initialize BPE tokenizer.
        """

        self.vocab_size = vocab_size
        self.vocab = []
        self.merges = []  # List of (pair, new_token) merges
        self.token_to_id = {}
        self.id_to_token = {}

    def _get_word_tokens(self, word: str) -> List[str]:
        """
        Convert word to list of characters with end-of-word marker.
        """

        if not word:
            return []

        tokens = list(word)
        tokens[-1] += Tokenizer.TOK_EOW  # Mark end of word
        return tokens

    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """
        Get all adjacent pairs from word tokens.
        """

        pairs = set()
        for i in range(len(word_tokens) - 1):
            pairs.add((word_tokens[i], word_tokens[i + 1]))
        return pairs

    def train(self, corpus: List[str], vocab_size: int = None) -> None:
        """
        Train BPE on corpus to learn merge rules.
        """

        if vocab_size:
            self.vocab_size = vocab_size

        # Count word frequencies and initialize character vocabulary
        word_freq = Counter(corpus)
        vocab = set()
        word_tokens = {}

        for word in word_freq:
            tokens = self._get_word_tokens(word)
            word_tokens[word] = tokens
            vocab.update(tokens)

        self.vocab = sorted(vocab)
        if Tokenizer.TOK_UNKNOWN not in vocab:
            self.vocab = [Tokenizer.TOK_UNKNOWN] + self.vocab

        # Greedy merge loop: count pairs, merge best, repeat
        self.merges = []

        while len(self.vocab) < self.vocab_size:
            pair_counts = _count_byte_pairs(word_tokens, word_freq)
            if not pair_counts:
                break

            best_pair = pair_counts.most_common(1)[0][0]
            merged_token = _merge_pair(word_tokens, best_pair)
            self.vocab.append(merged_token)
            self.merges.append(best_pair)

        self._build_mappings()

    def _build_mappings(self):
        """Build token-to-ID and ID-to-token mappings."""

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocab)}

    def _apply_merges(self, tokens: List[str]) -> List[str]:
        """
        Apply learned merge rules to token sequence.
        """

        if not self.merges:
            return tokens

        for merge_pair in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if (
                    i < len(tokens) - 1
                    and tokens[i] == merge_pair[0]
                    and tokens[i + 1] == merge_pair[1]
                ):
                    # Apply merge
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        return tokens

    def encode(self, text: str) -> List[int]:
        """
        Encode text using BPE.
        """

        if not self.vocab:
            return []

        # Simple word splitting (could be more sophisticated)
        words = text.split()
        all_tokens = []

        for word in words:
            # Get character-level tokens
            word_tokens = self._get_word_tokens(word)

            # Apply BPE merges
            merged_tokens = self._apply_merges(word_tokens)

            all_tokens.extend(merged_tokens)

        # Convert to IDs
        token_ids = []
        for token in all_tokens:
            token_ids.append(self.token_to_id.get(token, 0))  # 0 = <UNK>

        return token_ids

    def decode(self, tokens: List[int]) -> str:
        """
        Decode token IDs back to text.
        """

        if not self.id_to_token:
            return ""

        # Convert IDs to tokens
        token_strings = []
        for token_id in tokens:
            token = self.id_to_token.get(token_id, Tokenizer.TOK_UNKNOWN)
            token_strings.append(token)

        # Join and clean up
        text = "".join(token_strings)

        # Replace end-of-word markers with spaces
        text = text.replace(Tokenizer.TOK_EOW, " ")

        # Clean up extra spaces
        text = " ".join(text.split())

        return text


def create_tokenizer(
    strategy: str = "char", vocab_size: int = 1000, corpus: List[str] = None
) -> Tokenizer:
    """
    Factory function to create and train tokenizers.
    """

    if strategy == "char":
        tokenizer = CharTokenizer()
        if corpus:
            tokenizer.build_vocab(corpus)
    elif strategy == "bpe":
        tokenizer = BPETokenizer(vocab_size=vocab_size)
        if corpus:
            tokenizer.train(corpus, vocab_size)
    else:
        raise ValueError(
            f"Unknown tokenization strategy: '{strategy}'\n"
            f"  ❌ Strategy '{strategy}' is not recognized\n"
            f"  💡 TinyTorch supports 'char' (character-level) and 'bpe' (byte-pair encoding) strategies\n"
            f"  🔧 Use: create_tokenizer('char', corpus=texts) or create_tokenizer('bpe', vocab_size=1000, corpus=texts)"
        )

    return tokenizer


# | export
def tokenize_dataset(
    texts: List[str], tokenizer: Tokenizer, max_length: int = None
) -> List[List[int]]:
    """
    Tokenize a dataset with optional length limits.
    """

    tokenized = []
    for text in texts:
        tokens = tokenizer.encode(text)

        # Apply length limit
        if max_length and len(tokens) > max_length:
            tokens = tokens[:max_length]

        tokenized.append(tokens)

    return tokenized


# | export
def analyze_tokenization(texts: List[str], tokenizer: Tokenizer) -> Dict[str, float]:
    """
    Analyze tokenization statistics.
    """

    all_tokens = []
    total_chars = 0

    for text in texts:
        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_chars += len(text)

    # Calculate statistics
    tokenized_lengths = [len(tokenizer.encode(text)) for text in texts]

    stats = {
        "vocab_size": tokenizer.vocab_size,
        "avg_sequence_length": np.mean(tokenized_lengths),
        "max_sequence_length": max(tokenized_lengths) if tokenized_lengths else 0,
        "total_tokens": len(all_tokens),
        "compression_ratio": total_chars / len(all_tokens) if all_tokens else 0,
        "unique_tokens": len(set(all_tokens)),
    }

    return stats
