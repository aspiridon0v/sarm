# tests/test_tokenizer.py
"""Tests for the standalone CLIP tokenizer."""

import numpy as np
import open_clip
import pytest

from sarm.utils.tokenizer import load_tokenizer


@pytest.fixture(scope="session")
def openclip_tokenizer():
    """Load the open_clip tokenizer for comparison."""
    return open_clip.get_tokenizer("ViT-B-32")


@pytest.fixture(scope="session")
def custom_tokenizer():
    """Load our custom tokenizer."""
    return load_tokenizer()


def test_tokenizer_special_tokens(custom_tokenizer):
    """Test that special tokens are correctly defined."""
    assert custom_tokenizer.sot_token == 49406  # <|startoftext|>
    assert custom_tokenizer.eot_token == 49407  # <|endoftext|>
    assert custom_tokenizer.context_length == 77


def test_tokenizer_single_text(openclip_tokenizer, custom_tokenizer):
    """Test tokenization of a single text string."""
    text = "a photo of a cat"

    # Tokenize with both tokenizers
    openclip_tokens = openclip_tokenizer([text]).numpy()
    custom_tokens = custom_tokenizer([text])

    # Should produce identical results
    assert np.array_equal(
        openclip_tokens, custom_tokens
    ), f"Token mismatch:\nOpenCLIP: {openclip_tokens[0][:20]}\nCustom:   {custom_tokens[0][:20]}"


def test_tokenizer_multiple_texts(openclip_tokenizer, custom_tokenizer):
    """Test tokenization of multiple texts."""
    texts = [
        "a photo of a cat",
        "a dog playing in the park",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence and machine learning",
    ]

    # Tokenize with both tokenizers
    openclip_tokens = openclip_tokenizer(texts).numpy()
    custom_tokens = custom_tokenizer(texts)

    # Should produce identical results
    assert np.array_equal(
        openclip_tokens, custom_tokens
    ), "Tokens don't match for multiple texts"


def test_tokenizer_empty_string(openclip_tokenizer, custom_tokenizer):
    """Test tokenization of an empty string."""
    text = ""

    openclip_tokens = openclip_tokenizer([text]).numpy()
    custom_tokens = custom_tokenizer([text])

    assert np.array_equal(openclip_tokens, custom_tokens)


def test_tokenizer_special_characters(openclip_tokenizer, custom_tokenizer):
    """Test tokenization with special characters."""
    texts = [
        "Hello, world!",
        "It's a beautiful day.",
        "What's the meaning of life?",
        "I don't know...",
        "100% amazing!!!",
    ]

    openclip_tokens = openclip_tokenizer(texts).numpy()
    custom_tokens = custom_tokenizer(texts)

    assert np.array_equal(openclip_tokens, custom_tokens)


def test_tokenizer_long_text(openclip_tokenizer, custom_tokenizer):
    """Test tokenization with text longer than context length."""
    # Create a very long text that will exceed 77 tokens
    text = "This is a very long sentence. " * 20

    openclip_tokens = openclip_tokenizer([text]).numpy()
    custom_tokens = custom_tokenizer([text])

    # Should both truncate and match
    assert np.array_equal(openclip_tokens, custom_tokens)
    assert custom_tokens.shape[1] == 77


def test_tokenizer_unicode(openclip_tokenizer, custom_tokenizer):
    """Test tokenization with unicode characters."""
    texts = [
        "café",
        "naïve",
        "résumé",
        "jalapeño",
    ]

    openclip_tokens = openclip_tokenizer(texts).numpy()
    custom_tokens = custom_tokenizer(texts)

    assert np.array_equal(openclip_tokens, custom_tokens)


def test_tokenizer_numbers(openclip_tokenizer, custom_tokenizer):
    """Test tokenization with numbers."""
    texts = [
        "123",
        "1,000,000",
        "3.14159",
        "one two three",
    ]

    openclip_tokens = openclip_tokenizer(texts).numpy()
    custom_tokens = custom_tokenizer(texts)

    assert np.array_equal(openclip_tokens, custom_tokens)


def test_tokenizer_mixed_case(openclip_tokenizer, custom_tokenizer):
    """Test tokenization with mixed case (should be case-insensitive)."""
    texts = [
        "Hello World",
        "HELLO WORLD",
        "hello world",
        "HeLLo WoRLd",
    ]

    openclip_tokens = openclip_tokenizer(texts).numpy()
    custom_tokens = custom_tokenizer(texts)

    assert np.array_equal(openclip_tokens, custom_tokens)


def test_tokenizer_padding(custom_tokenizer):
    """Test that padding is applied correctly."""
    text = "short"

    tokens = custom_tokenizer([text])

    # Check that tokens are padded with zeros
    assert tokens.shape == (1, 77)

    # Find where EOT token is
    eot_pos = np.where(tokens[0] == custom_tokenizer.eot_token)[0][0]

    # Everything after EOT should be padding (zeros)
    assert np.all(tokens[0, eot_pos + 1 :] == 0)


def test_tokenizer_sot_eot_positions(custom_tokenizer):
    """Test that SOT and EOT tokens are in the correct positions."""
    text = "a cat"

    tokens = custom_tokenizer([text])[0]

    # First token should be SOT
    assert tokens[0] == custom_tokenizer.sot_token

    # Find EOT token
    eot_pos = np.where(tokens == custom_tokenizer.eot_token)[0][0]
    assert eot_pos > 0  # EOT should not be at position 0


def test_tokenizer_decode(custom_tokenizer):
    """Test that decoding works correctly."""
    text = "a photo of a cat"

    # Encode
    tokens = custom_tokenizer([text])[0]

    # Remove SOT, EOT, and padding
    sot_idx = 1  # Skip SOT
    eot_idx = np.where(tokens == custom_tokenizer.eot_token)[0][0]
    content_tokens = tokens[sot_idx:eot_idx]

    # Decode
    decoded = custom_tokenizer.decode(content_tokens.tolist())

    # Should roughly match original (may have spacing differences)
    assert "photo" in decoded.lower()
    assert "cat" in decoded.lower()


def test_tokenizer_consistency(custom_tokenizer):
    """Test that tokenizing the same text twice gives the same result."""
    text = "consistency test"

    tokens1 = custom_tokenizer([text])
    tokens2 = custom_tokenizer([text])

    assert np.array_equal(tokens1, tokens2)


def test_tokenizer_batch_consistency(custom_tokenizer):
    """Test that batching doesn't affect tokenization."""
    texts = ["first", "second", "third"]

    # Tokenize together
    batch_tokens = custom_tokenizer(texts)

    # Tokenize separately
    separate_tokens = np.stack([custom_tokenizer([t])[0] for t in texts])

    assert np.array_equal(batch_tokens, separate_tokens)


def test_tokenizer_vocab_size(custom_tokenizer):
    """Test that vocabulary size is correct."""
    # CLIP uses 49408 vocab size
    assert len(custom_tokenizer.encoder) == 49408
    assert len(custom_tokenizer.decoder) == 49408


def test_tokenizer_real_world_examples(openclip_tokenizer, custom_tokenizer):
    """Test with real-world example captions."""
    texts = [
        "A person standing in the rain with an umbrella",
        "Two dogs running through a field of flowers",
        "A beautiful sunset over the ocean",
        "A close-up of a red apple on a wooden table",
        "An astronaut floating in space with Earth in the background",
    ]

    openclip_tokens = openclip_tokenizer(texts).numpy()
    custom_tokens = custom_tokenizer(texts)

    assert np.array_equal(openclip_tokens, custom_tokens)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
