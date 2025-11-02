# src/sarm/utils/tokenizer.py
"""
CLIP tokenizer implementation without external dependencies.
Based on OpenAI's CLIP tokenizer which uses BPE (Byte Pair Encoding).
"""

import gzip
import html
import os
from functools import lru_cache
from typing import List, Union

import ftfy
import numpy as np
import regex as re


@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a significant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    """Basic text cleaning."""
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    """Clean up whitespace."""
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


class ClipTokenizer:
    """Simple CLIP tokenizer using BPE."""

    def __init__(self, bpe_path: str = None):
        """
        Initialize the tokenizer.

        Args:
            bpe_path: Path to BPE vocab file. If None, will try to download from OpenAI.
        """
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

        # Load BPE merges
        if bpe_path is None:
            bpe_path = self._download_bpe_vocab()

        with (
            gzip.open(bpe_path, "rt", encoding="utf-8")
            if bpe_path.endswith(".gz")
            else open(bpe_path, "r", encoding="utf-8")
        ) as f:
            merges = f.read().split("\n")

        merges = merges[
            1 : 49152 - 256 - 2 + 1
        ]  # Skip header, get 49152-256-2=48894 merges
        merges = [tuple(merge.split()) for merge in merges]

        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v + "</w>" for v in vocab]
        for merge in merges:
            vocab.append("".join(merge))
        vocab.extend(["<|startoftext|>", "<|endoftext|>"])

        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {
            "<|startoftext|>": "<|startoftext|>",
            "<|endoftext|>": "<|endoftext|>",
        }

        # Use regex pattern that matches CLIP tokenizer
        self.pat = re.compile(
            r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            re.IGNORECASE,
        )

        # Special tokens
        self.sot_token = self.encoder["<|startoftext|>"]
        self.eot_token = self.encoder["<|endoftext|>"]
        self.context_length = 77

    def _download_bpe_vocab(self):
        """Download BPE vocab from OpenAI if not already present."""
        import urllib.request

        cache_dir = os.path.expanduser("~/.cache/clip")
        os.makedirs(cache_dir, exist_ok=True)

        bpe_path = os.path.join(cache_dir, "bpe_simple_vocab_16e6.txt.gz")

        if not os.path.exists(bpe_path):
            url = "https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz"
            print(f"Downloading BPE vocab from {url}...")
            urllib.request.urlretrieve(url, bpe_path)
            print(f"Saved to {bpe_path}")

        return bpe_path

    def bpe(self, token):
        """Apply BPE to token."""
        if token in self.cache:
            return self.cache[token]

        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        """Encode text to BPE token IDs."""
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = "".join(self.byte_encoder[b] for b in token.encode("utf-8"))
            bpe_tokens.extend(
                self.encoder[bpe_token] for bpe_token in self.bpe(token).split(" ")
            )
        return bpe_tokens

    def decode(self, tokens):
        """Decode BPE token IDs to text."""
        text = "".join([self.decoder[token] for token in tokens])
        text = (
            bytearray([self.byte_decoder[c] for c in text])
            .decode("utf-8", errors="replace")
            .replace("</w>", " ")
        )
        return text

    def __call__(
        self, texts: Union[str, List[str]], context_length: int = None
    ) -> np.ndarray:
        """
        Tokenize text(s) and return padded token arrays.

        Args:
            texts: Single string or list of strings to tokenize
            context_length: Maximum sequence length (default: 77)

        Returns:
            numpy array of shape (len(texts), context_length) with token IDs
        """
        if context_length is None:
            context_length = self.context_length

        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.sot_token
        eot_token = self.eot_token
        all_tokens = []

        for text in texts:
            tokens = [sot_token] + self.encode(text) + [eot_token]
            result = np.zeros(context_length, dtype=np.int32)

            if len(tokens) > context_length:
                # Truncate
                tokens = tokens[:context_length]
                tokens[-1] = eot_token

            result[: len(tokens)] = tokens
            all_tokens.append(result)

        return np.stack(all_tokens)


def load_tokenizer(bpe_path: str = None) -> ClipTokenizer:
    """
    Load the CLIP tokenizer.

    Args:
        bpe_path: Optional path to BPE vocab file. If None, downloads from OpenAI.

    Returns:
        ClipTokenizer instance
    """
    return ClipTokenizer(bpe_path=bpe_path)
