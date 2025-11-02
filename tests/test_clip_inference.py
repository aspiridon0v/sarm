#!/usr/bin/env python3
"""
Test CLIP inference by comparing sarm (JAX) implementation with OpenCLIP (PyTorch).
Tests both correctness (right text selected) and equivalence (similar outputs).
"""

import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import open_clip
import pytest
import torch
from PIL import Image, ImageDraw

from sarm.model.clip import CLIP, load_clip_npz
from sarm.utils.convert_clip import main as export_clip_weights
from sarm.utils.tokenizer import load_tokenizer

WEIGHTS_PATH = "checkpoints/clip_vit_b32_openai.npz"

# Check for GPU availability
HAS_CUDA = torch.cuda.is_available()
try:
    HAS_JAX_GPU = len(jax.devices("gpu")) > 0
except RuntimeError:
    HAS_JAX_GPU = False


@pytest.fixture(scope="session")
def ensure_weights():
    """Export weights once per session if not already present."""
    if not os.path.exists(WEIGHTS_PATH):
        export_clip_weights()
    assert os.path.exists(WEIGHTS_PATH), "Failed to export CLIP weights to .npz"
    return WEIGHTS_PATH


@pytest.fixture(scope="module")
def openclip_model(torch_device):
    """Load OpenCLIP model with preprocessing."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    model.eval()
    model.to(torch_device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer, preprocess, torch_device


@pytest.fixture(scope="module")
def sarm_model(ensure_weights):
    """Load sarm CLIP model."""
    model = CLIP(
        image_size=224,
        patch_size=32,
        vision_width=768,
        vision_layers=12,
        vision_heads=12,
        context_length=77,
        vocab_size=49408,
        text_width=512,
        text_layers=12,
        text_heads=8,
        embed_dim=512,
    )
    model = load_clip_npz(model, ensure_weights)
    tokenizer = load_tokenizer()
    return model, tokenizer


@pytest.fixture(scope="module")
def test_image_and_texts():
    """Create test image and text descriptions."""
    # Create a simple test image (red square on gray background)
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    d = ImageDraw.Draw(img)
    d.rectangle([64, 64, 160, 160], fill=(255, 0, 0))

    # Test captions
    texts = [
        "a red square",
        "a blue circle",
        "a gray background",
        "a red square on a gray background",
        "a photograph of nature",
    ]

    return img, texts


def preprocess_image_sarm(image: Image.Image) -> jax.Array:
    """
    Preprocess an image for CLIP (sarm/JAX version).

    Args:
        image: PIL Image

    Returns:
        Preprocessed image tensor (3, 224, 224)
    """
    # Resize to 224x224
    image = image.resize((224, 224), Image.BICUBIC)

    # Convert to numpy array and normalize
    image = np.array(image).astype(np.float32) / 255.0

    # CLIP normalization
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])

    # Normalize each channel
    image = (image - mean) / std

    # Convert from HWC to CHW
    image = np.transpose(image, (2, 0, 1))

    return jnp.array(image)


def test_sarm_inference_selects_correct_text(sarm_model, test_image_and_texts):
    """Test that sarm model selects the correct text with high probability."""
    model, tokenizer = sarm_model
    img, texts = test_image_and_texts

    # Preprocess image
    image_tensor = preprocess_image_sarm(img)

    # Tokenize texts
    text_tokens = tokenizer(texts)

    # Encode image
    image_features = model.encode_image(image_tensor)
    image_features = image_features / jnp.linalg.norm(image_features)

    # Encode texts using vmap
    text_tokens_jax = jnp.array(text_tokens)
    text_features = jax.vmap(model.encode_text)(text_tokens_jax)
    # Normalize each text feature
    text_features = text_features / jnp.linalg.norm(text_features, axis=1, keepdims=True)

    # Compute similarities
    similarities = image_features @ text_features.T
    similarities = np.array(similarities)

    # Convert to probabilities
    probs = np.exp(similarities * 100) / np.sum(np.exp(similarities * 100))

    # Find best match
    best_idx = np.argmax(similarities)
    best_text = texts[best_idx]
    best_prob = probs[best_idx]

    # The best match should be "a red square on a gray background" (index 3)
    # or at least "a red square" (index 0)
    assert best_idx in [0, 3], (
        f"Expected best match to be 'a red square' or 'a red square on a gray background', "
        f"but got '{best_text}' (index {best_idx})"
    )

    # The probability should be reasonably high (>30%)
    assert best_prob > 0.3, (
        f"Best match probability too low: {best_prob:.2%}. "
        f"Expected at least 30% for '{best_text}'"
    )

    print(f"\nSARM model results:")
    print(
        f"  Best match: '{best_text}' (similarity: {similarities[best_idx]:.4f}, prob: {best_prob:.2%})"
    )
    for i, text in enumerate(texts):
        print(f"    {text:50s} | Prob: {probs[i]:.2%}")


def test_openclip_inference_selects_correct_text(openclip_model, test_image_and_texts):
    """Test that OpenCLIP model selects the correct text with high probability."""
    model, tokenizer, preprocess, torch_device = openclip_model
    img, texts = test_image_and_texts

    # Preprocess image and texts
    image_tensor = preprocess(img).unsqueeze(0).to(torch_device)
    text_tokens = tokenizer(texts).to(torch_device)

    # Encode
    with torch.no_grad():
        image_features = model.encode_image(image_tensor)
        text_features = model.encode_text(text_tokens)

        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarities
        similarities = (image_features @ text_features.T).squeeze(0)
        similarities = similarities.cpu().numpy()

    # Convert to probabilities
    probs = np.exp(similarities * 100) / np.sum(np.exp(similarities * 100))

    # Find best match
    best_idx = np.argmax(similarities)
    best_text = texts[best_idx]
    best_prob = probs[best_idx]

    # The best match should be "a red square on a gray background" (index 3)
    # or at least "a red square" (index 0)
    assert best_idx in [0, 3], (
        f"Expected best match to be 'a red square' or 'a red square on a gray background', "
        f"but got '{best_text}' (index {best_idx})"
    )

    # The probability should be reasonably high (>30%)
    assert best_prob > 0.3, (
        f"Best match probability too low: {best_prob:.2%}. "
        f"Expected at least 30% for '{best_text}'"
    )

    print(f"\nOpenCLIP model results:")
    print(
        f"  Best match: '{best_text}' (similarity: {similarities[best_idx]:.4f}, prob: {best_prob:.2%})"
    )
    for i, text in enumerate(texts):
        print(f"    {text:50s} | Prob: {probs[i]:.2%}")


def test_sarm_vs_openclip_equivalence(sarm_model, openclip_model, test_image_and_texts):
    """
    Test that sarm and OpenCLIP produce similar outputs for the same inputs.
    This ensures both models produce equivalent results.
    """
    sarm_clip, sarm_tokenizer = sarm_model
    openclip, openclip_tokenizer, preprocess, torch_device = openclip_model
    img, texts = test_image_and_texts

    # === SARM inference ===
    # Preprocess image
    sarm_image = preprocess_image_sarm(img)

    # Tokenize texts
    sarm_text_tokens = sarm_tokenizer(texts)

    # Encode image
    sarm_image_features = sarm_clip.encode_image(sarm_image)
    sarm_image_features = sarm_image_features / jnp.linalg.norm(sarm_image_features)

    # Encode texts using vmap
    sarm_text_tokens_jax = jnp.array(sarm_text_tokens)
    sarm_text_features = jax.vmap(sarm_clip.encode_text)(sarm_text_tokens_jax)
    # Normalize each text feature
    sarm_text_features = sarm_text_features / jnp.linalg.norm(
        sarm_text_features, axis=1, keepdims=True
    )

    # Compute similarities
    sarm_similarities = sarm_image_features @ sarm_text_features.T
    sarm_similarities = np.array(sarm_similarities)

    # === OpenCLIP inference ===
    # Preprocess image and texts
    openclip_image = preprocess(img).unsqueeze(0).to(torch_device)
    openclip_text_tokens = openclip_tokenizer(texts).to(torch_device)

    # Encode
    with torch.no_grad():
        openclip_image_features = openclip.encode_image(openclip_image)
        openclip_text_features = openclip.encode_text(openclip_text_tokens)

        # Normalize
        openclip_image_features = openclip_image_features / openclip_image_features.norm(
            dim=-1, keepdim=True
        )
        openclip_text_features = openclip_text_features / openclip_text_features.norm(
            dim=-1, keepdim=True
        )

        # Compute similarities
        openclip_similarities = (openclip_image_features @ openclip_text_features.T).squeeze(0)
        openclip_similarities = openclip_similarities.cpu().numpy()

    # Convert features to numpy for comparison
    sarm_image_features_np = np.array(sarm_image_features)
    sarm_text_features_np = np.array(sarm_text_features)
    openclip_image_features_np = openclip_image_features.squeeze(0).cpu().numpy()
    openclip_text_features_np = openclip_text_features.cpu().numpy()

    # === Verify both models select the same text ===
    sarm_best_idx = np.argmax(sarm_similarities)
    openclip_best_idx = np.argmax(openclip_similarities)

    assert sarm_best_idx == openclip_best_idx, (
        f"Models selected different texts: "
        f"sarm selected '{texts[sarm_best_idx]}' (index {sarm_best_idx}), "
        f"OpenCLIP selected '{texts[openclip_best_idx]}' (index {openclip_best_idx})"
    )

    # === Verify image features are similar ===
    image_max_diff = np.max(np.abs(sarm_image_features_np - openclip_image_features_np))
    image_mean_diff = np.mean(np.abs(sarm_image_features_np - openclip_image_features_np))

    assert image_max_diff < 1e-4, (
        f"Image features differ too much between models: "
        f"max diff = {image_max_diff:.3e}, mean diff = {image_mean_diff:.3e}"
    )
    assert image_mean_diff < 2e-5, f"Image features mean difference too high: {image_mean_diff:.3e}"

    # === Verify text features are similar ===
    text_max_diff = np.max(np.abs(sarm_text_features_np - openclip_text_features_np))
    text_mean_diff = np.mean(np.abs(sarm_text_features_np - openclip_text_features_np))

    assert text_max_diff < 1e-4, (
        f"Text features differ too much between models: "
        f"max diff = {text_max_diff:.3e}, mean diff = {text_mean_diff:.3e}"
    )
    assert text_mean_diff < 2e-5, f"Text features mean difference too high: {text_mean_diff:.3e}"

    # === Verify similarities are close ===
    similarity_max_diff = np.max(np.abs(sarm_similarities - openclip_similarities))
    similarity_mean_diff = np.mean(np.abs(sarm_similarities - openclip_similarities))

    assert similarity_max_diff < 1e-4, (
        f"Similarity scores differ too much between models: "
        f"max diff = {similarity_max_diff:.3e}, mean diff = {similarity_mean_diff:.3e}"
    )
    assert (
        similarity_mean_diff < 2e-5
    ), f"Similarity scores mean difference too high: {similarity_mean_diff:.3e}"

    print(f"\nEquivalence test results:")
    print(f"  Both models selected: '{texts[sarm_best_idx]}'")
    print(f"  Image features - max diff: {image_max_diff:.3e}, mean diff: {image_mean_diff:.3e}")
    print(f"  Text features - max diff: {text_max_diff:.3e}, mean diff: {text_mean_diff:.3e}")
    print(
        f"  Similarities - max diff: {similarity_max_diff:.3e}, mean diff: {similarity_mean_diff:.3e}"
    )


@pytest.mark.skipif(not (HAS_CUDA and HAS_JAX_GPU), reason="GPU not available")
def test_sarm_jit_gpu_inference_selects_correct_text(sarm_model, test_image_and_texts):
    """Test that JIT-compiled sarm model on GPU selects the correct text with high probability."""
    model, tokenizer = sarm_model
    img, texts = test_image_and_texts

    # JIT compile the encode functions using filter_jit for Equinox modules
    jit_encode_image = eqx.filter_jit(model.encode_image)
    jit_encode_text = eqx.filter_jit(eqx.filter_vmap(model.encode_text))

    # Preprocess image
    image_tensor = preprocess_image_sarm(img)

    # Tokenize texts
    text_tokens = tokenizer(texts)
    text_tokens_jax = jnp.array(text_tokens)

    # Encode image with JIT
    image_features = jit_encode_image(image_tensor)
    image_features = image_features / jnp.linalg.norm(image_features)

    # Encode texts with JIT
    text_features = jit_encode_text(text_tokens_jax)
    # Normalize each text feature
    text_features = text_features / jnp.linalg.norm(text_features, axis=1, keepdims=True)

    # Compute similarities
    similarities = image_features @ text_features.T
    similarities = np.array(similarities)

    # Convert to probabilities
    probs = np.exp(similarities * 100) / np.sum(np.exp(similarities * 100))

    # Find best match
    best_idx = np.argmax(similarities)
    best_text = texts[best_idx]
    best_prob = probs[best_idx]

    # The best match should be "a red square on a gray background" (index 3)
    # or at least "a red square" (index 0)
    assert best_idx in [0, 3], (
        f"Expected best match to be 'a red square' or 'a red square on a gray background', "
        f"but got '{best_text}' (index {best_idx})"
    )

    # The probability should be reasonably high (>30%)
    assert best_prob > 0.3, (
        f"Best match probability too low: {best_prob:.2%}. "
        f"Expected at least 30% for '{best_text}'"
    )

    print(f"\nSARM JIT GPU model results:")
    print(
        f"  Best match: '{best_text}' (similarity: {similarities[best_idx]:.4f}, prob: {best_prob:.2%})"
    )
    for i, text in enumerate(texts):
        print(f"    {text:50s} | Prob: {probs[i]:.2%}")


@pytest.mark.skipif(not (HAS_CUDA and HAS_JAX_GPU), reason="GPU not available")
def test_sarm_jit_gpu_vs_openclip_gpu_equivalence(sarm_model, openclip_model, test_image_and_texts):
    """
    Test that JIT-compiled sarm model on GPU and OpenCLIP on GPU produce similar outputs.
    This ensures both models produce equivalent results when running on GPU with JIT.
    """
    sarm_clip, sarm_tokenizer = sarm_model
    openclip, openclip_tokenizer, preprocess, torch_device = openclip_model
    img, texts = test_image_and_texts

    # JIT compile the encode functions for sarm using filter_jit for Equinox modules
    jit_encode_image = eqx.filter_jit(sarm_clip.encode_image)
    jit_encode_text = eqx.filter_jit(eqx.filter_vmap(sarm_clip.encode_text))

    # === SARM JIT GPU inference ===
    # Preprocess image
    sarm_image = preprocess_image_sarm(img)

    # Tokenize texts
    sarm_text_tokens = sarm_tokenizer(texts)
    sarm_text_tokens_jax = jnp.array(sarm_text_tokens)

    # Encode image with JIT
    sarm_image_features = jit_encode_image(sarm_image)
    sarm_image_features = sarm_image_features / jnp.linalg.norm(sarm_image_features)

    # Encode texts with JIT
    sarm_text_features = jit_encode_text(sarm_text_tokens_jax)
    # Normalize each text feature
    sarm_text_features = sarm_text_features / jnp.linalg.norm(
        sarm_text_features, axis=1, keepdims=True
    )

    # Compute similarities
    sarm_similarities = sarm_image_features @ sarm_text_features.T
    sarm_similarities = np.array(sarm_similarities)

    # === OpenCLIP GPU inference ===
    # Preprocess image and texts (model already on GPU from fixture)
    openclip_image = preprocess(img).unsqueeze(0).to(torch_device)
    openclip_text_tokens = openclip_tokenizer(texts).to(torch_device)

    # Encode
    with torch.no_grad():
        openclip_image_features = openclip.encode_image(openclip_image)
        openclip_text_features = openclip.encode_text(openclip_text_tokens)

        # Normalize
        openclip_image_features = openclip_image_features / openclip_image_features.norm(
            dim=-1, keepdim=True
        )
        openclip_text_features = openclip_text_features / openclip_text_features.norm(
            dim=-1, keepdim=True
        )

        # Compute similarities
        openclip_similarities = (openclip_image_features @ openclip_text_features.T).squeeze(0)
        openclip_similarities = openclip_similarities.cpu().numpy()

    # Convert features to numpy for comparison
    sarm_image_features_np = np.array(sarm_image_features)
    sarm_text_features_np = np.array(sarm_text_features)
    openclip_image_features_np = openclip_image_features.squeeze(0).cpu().numpy()
    openclip_text_features_np = openclip_text_features.cpu().numpy()

    # === Verify both models select the same text ===
    sarm_best_idx = np.argmax(sarm_similarities)
    openclip_best_idx = np.argmax(openclip_similarities)

    assert sarm_best_idx == openclip_best_idx, (
        f"Models selected different texts: "
        f"sarm selected '{texts[sarm_best_idx]}' (index {sarm_best_idx}), "
        f"OpenCLIP selected '{texts[openclip_best_idx]}' (index {openclip_best_idx})"
    )

    # === Verify image features are similar ===
    image_max_diff = np.max(np.abs(sarm_image_features_np - openclip_image_features_np))
    image_mean_diff = np.mean(np.abs(sarm_image_features_np - openclip_image_features_np))

    assert image_max_diff < 1e-4, (
        f"Image features differ too much between models: "
        f"max diff = {image_max_diff:.3e}, mean diff = {image_mean_diff:.3e}"
    )
    assert image_mean_diff < 2e-5, f"Image features mean difference too high: {image_mean_diff:.3e}"

    # === Verify text features are similar ===
    text_max_diff = np.max(np.abs(sarm_text_features_np - openclip_text_features_np))
    text_mean_diff = np.mean(np.abs(sarm_text_features_np - openclip_text_features_np))

    assert text_max_diff < 1e-4, (
        f"Text features differ too much between models: "
        f"max diff = {text_max_diff:.3e}, mean diff = {text_mean_diff:.3e}"
    )
    assert text_mean_diff < 2e-5, f"Text features mean difference too high: {text_mean_diff:.3e}"

    # === Verify similarities are close ===
    similarity_max_diff = np.max(np.abs(sarm_similarities - openclip_similarities))
    similarity_mean_diff = np.mean(np.abs(sarm_similarities - openclip_similarities))

    assert similarity_max_diff < 1e-4, (
        f"Similarity scores differ too much between models: "
        f"max diff = {similarity_max_diff:.3e}, mean diff = {similarity_mean_diff:.3e}"
    )
    assert (
        similarity_mean_diff < 2e-5
    ), f"Similarity scores mean difference too high: {similarity_mean_diff:.3e}"

    print(f"\nJIT GPU Equivalence test results:")
    print(f"  Both models selected: '{texts[sarm_best_idx]}'")
    print(f"  Image features - max diff: {image_max_diff:.3e}, mean diff: {image_mean_diff:.3e}")
    print(f"  Text features - max diff: {text_max_diff:.3e}, mean diff: {text_mean_diff:.3e}")
    print(
        f"  Similarities - max diff: {similarity_max_diff:.3e}, mean diff: {similarity_mean_diff:.3e}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
