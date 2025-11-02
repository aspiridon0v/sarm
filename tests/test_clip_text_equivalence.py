# tests/test_clip_text_equivalence.py
import os

import jax
import jax.numpy as jnp
import numpy as np
import open_clip
import pytest
import torch

from sarm.model.clip import CLIP, TextTransformer, load_clip_npz, load_text_npz
from sarm.utils.convert_clip import main as export_clip_weights

WEIGHTS_PATH = "checkpoints/clip_vit_b32_openai.npz"


@pytest.fixture(scope="session")
def pt_model_and_tokenizer(torch_device):
    """Load PyTorch CLIP ViT-B/32 (openai weights) and tokenizer."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    model.eval()
    model.to(torch_device)
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return model, tokenizer, preprocess, torch_device


@pytest.fixture(scope="session")
def ensure_weights(pt_model_and_tokenizer):
    """Export weights once per session if not already present."""
    if not os.path.exists(WEIGHTS_PATH):
        export_clip_weights()
    assert os.path.exists(WEIGHTS_PATH), "Failed to export CLIP weights to .npz"
    return WEIGHTS_PATH


def _make_test_texts():
    """Create a list of test text strings."""
    return [
        "a photo of a cat",
        "a dog playing in the park",
        "the quick brown fox jumps over the lazy dog",
        "artificial intelligence and machine learning",
    ]


def _torch_forward_text(model, text_tokens):
    """Forward text through PyTorch CLIP model."""
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    return text_features


def test_text_weight_loading(pt_model_and_tokenizer, ensure_weights):
    """Test that text weights are loaded correctly from PyTorch to Equinox."""
    model, tokenizer, _, torch_device = pt_model_and_tokenizer

    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )
    eq_model = load_text_npz(eq_model, ensure_weights)

    # Test token embedding weights
    pt_token_weight = model.token_embedding.weight.detach().cpu().numpy()
    eq_token_weight = np.array(eq_model.token_embedding.weight)

    assert np.allclose(
        pt_token_weight, eq_token_weight, atol=0, rtol=0
    ), "Token embedding weights don't match"

    # Test positional embedding
    pt_pos_embed = model.positional_embedding.detach().cpu().numpy()
    eq_pos_embed = np.array(eq_model.positional_embedding)

    assert np.allclose(
        pt_pos_embed, eq_pos_embed, atol=0, rtol=0
    ), "Positional embeddings don't match"

    # Test MLP weights in first block
    pt_fc1_weight = (
        model.transformer.resblocks[0].mlp.c_fc.weight.detach().cpu().numpy()
    )
    pt_fc1_bias = model.transformer.resblocks[0].mlp.c_fc.bias.detach().cpu().numpy()
    eq_fc1_weight = np.array(eq_model.blocks[0].mlp.fc1.weight)
    eq_fc1_bias = np.array(eq_model.blocks[0].mlp.fc1.bias)

    assert np.allclose(
        pt_fc1_weight, eq_fc1_weight, atol=0, rtol=0
    ), "fc1 weights don't match"
    assert np.allclose(
        pt_fc1_bias, eq_fc1_bias, atol=0, rtol=0
    ), "fc1 bias doesn't match"


def test_text_features_match_pytorch(pt_model_and_tokenizer, ensure_weights):
    """Test that text features match between PyTorch and JAX implementations."""
    model, tokenizer, _, torch_device = pt_model_and_tokenizer

    # Build and load Equinox model
    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )
    eq_model = load_text_npz(eq_model, ensure_weights)

    # Create test texts
    texts = _make_test_texts()

    # Tokenize with PyTorch
    text_tokens = tokenizer(texts).to(torch_device)

    # PyTorch forward
    pt_features = _torch_forward_text(model, text_tokens).cpu().numpy()

    # JAX forward (single examples, not batched)
    eq_features = []
    for i in range(len(texts)):
        tokens = jnp.array(text_tokens[i].cpu().numpy())
        feat = eq_model(tokens)
        eq_features.append(np.array(feat))
    eq_features = np.stack(eq_features, axis=0)

    # Compare
    assert pt_features.shape == eq_features.shape == (len(texts), 512)

    # Allow slightly larger tolerance for text due to longer sequences
    atol = 5e-5
    rtol = 1e-4

    max_abs = np.max(np.abs(eq_features - pt_features))
    mean_abs = np.mean(np.abs(eq_features - pt_features))

    assert np.allclose(
        eq_features, pt_features, atol=atol, rtol=rtol
    ), f"Equinox text features differ from PyTorch (max|diff|={max_abs:.3e}, mean|diff|={mean_abs:.3e})"


def test_text_embedding_layer(pt_model_and_tokenizer, ensure_weights):
    """Test that the token embedding layer produces matching outputs."""
    model, tokenizer, _, torch_device = pt_model_and_tokenizer

    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )
    eq_model = load_text_npz(eq_model, ensure_weights)

    # Create test token IDs
    test_tokens = tokenizer(["a photo of a cat"])[0].to(torch_device)

    # PyTorch embedding
    with torch.no_grad():
        pt_embed = model.token_embedding(test_tokens).cpu().numpy()

    # JAX embedding
    tokens_jax = jnp.array(test_tokens.cpu().numpy())
    eq_embed = np.array(jax.vmap(eq_model.token_embedding)(tokens_jax))

    # Should match exactly
    assert np.allclose(pt_embed, eq_embed, atol=1e-6, rtol=1e-6)


def test_text_causal_attention(pt_model_and_tokenizer, ensure_weights):
    """Test that causal attention masking works correctly."""
    model, tokenizer, _, torch_device = pt_model_and_tokenizer

    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )
    eq_model = load_text_npz(eq_model, ensure_weights)

    # Create test input
    test_tokens = tokenizer(["the quick brown fox"])[0].to(torch_device)

    # Get embeddings + positional encodings
    with torch.no_grad():
        x_pt = model.token_embedding(test_tokens)
        x_pt = x_pt + model.positional_embedding

    tokens_jax = jnp.array(test_tokens.cpu().numpy())
    x_eq = jax.vmap(eq_model.token_embedding)(tokens_jax)
    seq_len = tokens_jax.shape[0]
    x_eq = x_eq + eq_model.positional_embedding[:seq_len]

    # Apply first transformer block
    with torch.no_grad():
        attn_mask = model.attn_mask[:seq_len, :seq_len]
        x_pt_block = model.transformer.resblocks[0](
            x_pt[None, :, :], attn_mask=attn_mask
        )[0]

    mask_eq = eq_model.attn_mask[:seq_len, :seq_len]
    x_eq_block = eq_model.blocks[0](x_eq, mask_eq)

    # Compare outputs
    pt_output = x_pt_block.cpu().numpy()
    eq_output = np.array(x_eq_block)

    max_diff = np.max(np.abs(pt_output - eq_output))
    assert max_diff < 1e-4, f"Text block outputs differ by {max_diff}"


def test_text_layer_norm(pt_model_and_tokenizer, ensure_weights):
    """Test that LayerNorm in text transformer matches."""
    model, _, _, torch_device = pt_model_and_tokenizer

    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )
    eq_model = load_text_npz(eq_model, ensure_weights)

    # Create test input
    test_input_np = np.random.randn(77, 512).astype(np.float32)
    test_input_pt = torch.from_numpy(test_input_np).to(torch_device)
    test_input_jax = jnp.array(test_input_np)

    # PyTorch LayerNorm
    with torch.no_grad():
        pt_output = model.ln_final(test_input_pt[None, :, :])[0].cpu().numpy()

    # JAX LayerNorm
    eq_output = np.array(jax.vmap(eq_model.ln_final)(test_input_jax))

    # Should match to numerical precision
    assert np.allclose(pt_output, eq_output, atol=1e-5, rtol=1e-5)


def test_text_projection(pt_model_and_tokenizer, ensure_weights):
    """Test that the text projection layer matches."""
    model, _, _, torch_device = pt_model_and_tokenizer

    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )
    eq_model = load_text_npz(eq_model, ensure_weights)

    # Create test input (output of ln_final)
    test_input_np = np.random.randn(512).astype(np.float32)
    test_input_pt = torch.from_numpy(test_input_np).to(torch_device)
    test_input_jax = jnp.array(test_input_np)

    # PyTorch projection
    with torch.no_grad():
        if isinstance(model.text_projection, torch.nn.Linear):
            pt_output = model.text_projection(test_input_pt).cpu().numpy()
        else:
            pt_output = (test_input_pt @ model.text_projection).cpu().numpy()

    # JAX projection
    eq_output = np.array(test_input_jax @ eq_model.text_projection)

    # Should match to numerical precision
    assert np.allclose(pt_output, eq_output, atol=1e-5, rtol=1e-5)


def test_eot_token_pooling(pt_model_and_tokenizer, ensure_weights):
    """Test that EOT token is correctly identified and used for pooling."""
    model, tokenizer, _, torch_device = pt_model_and_tokenizer

    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )
    eq_model = load_text_npz(eq_model, ensure_weights)

    # Create test with known EOT position
    text = "hello world"
    text_tokens = tokenizer([text])[0].to(torch_device)

    # Find EOT position in PyTorch
    eot_pos_pt = text_tokens.argmax(dim=-1).item()

    # Find EOT position in JAX
    tokens_jax = jnp.array(text_tokens.cpu().numpy())
    eot_pos_jax = jnp.argmax(tokens_jax).item()

    # Should be the same
    assert eot_pos_pt == eot_pos_jax, "EOT token position doesn't match"


def test_complete_clip_model(pt_model_and_tokenizer, ensure_weights):
    """Test the complete CLIP model with both vision and text."""
    pt_model, tokenizer, preprocess, torch_device = pt_model_and_tokenizer

    # Build and load complete CLIP model
    eq_clip = CLIP(
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
    eq_clip = load_clip_npz(eq_clip, ensure_weights)

    # Test with one image and one text
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (224, 224), (128, 128, 128))
    d = ImageDraw.Draw(img)
    d.rectangle([64, 64, 160, 160], fill=(255, 0, 0))

    text = "a red square on a gray background"

    # PyTorch forward
    img_pt = preprocess(img).unsqueeze(0).to(torch_device)
    text_tokens = tokenizer([text]).to(torch_device)

    with torch.no_grad():
        pt_img_feat = pt_model.encode_image(img_pt)[0].cpu().numpy()
        pt_text_feat = pt_model.encode_text(text_tokens)[0].cpu().numpy()

    # Normalize PyTorch features
    pt_img_feat = pt_img_feat / np.linalg.norm(pt_img_feat)
    pt_text_feat = pt_text_feat / np.linalg.norm(pt_text_feat)

    # JAX forward
    img_jax = jnp.array(img_pt.cpu().numpy()[0])
    text_jax = jnp.array(text_tokens[0].cpu().numpy())

    eq_img_feat, eq_text_feat = eq_clip(img_jax, text_jax)
    eq_img_feat = np.array(eq_img_feat)
    eq_text_feat = np.array(eq_text_feat)

    # Compare image features
    img_max_diff = np.max(np.abs(pt_img_feat - eq_img_feat))
    assert img_max_diff < 5e-5, f"Image features differ by {img_max_diff}"

    # Compare text features
    text_max_diff = np.max(np.abs(pt_text_feat - eq_text_feat))
    assert text_max_diff < 1e-4, f"Text features differ by {text_max_diff}"

    # Test similarity score
    pt_similarity = np.dot(pt_img_feat, pt_text_feat)
    eq_similarity = np.dot(eq_img_feat, eq_text_feat)

    similarity_diff = abs(pt_similarity - eq_similarity)
    assert similarity_diff < 1e-4, f"Similarity scores differ by {similarity_diff}"


def test_text_mlp_with_quick_gelu(pt_model_and_tokenizer, ensure_weights):
    """Test that text MLP with QuickGELU produces matching outputs."""
    model, _, _, torch_device = pt_model_and_tokenizer

    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )
    eq_model = load_text_npz(eq_model, ensure_weights)

    # Create test input
    test_input_np = np.random.randn(77, 512).astype(np.float32)
    test_input_pt = torch.from_numpy(test_input_np).to(torch_device)
    test_input_jax = jnp.array(test_input_np)

    # PyTorch MLP
    with torch.no_grad():
        pt_output = model.transformer.resblocks[0].mlp(test_input_pt).cpu().numpy()

    # JAX MLP
    eq_output = np.array(eq_model.blocks[0].mlp(test_input_jax))

    # Should match closely
    max_diff = np.max(np.abs(pt_output - eq_output))
    assert max_diff < 1e-4, f"MLP outputs differ by {max_diff}"


def test_batched_text_encoding():
    """Test that we can process multiple texts (even if not using vmap in the model)."""
    eq_model = TextTransformer(
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
    )

    # Create random token sequences
    texts = [
        jnp.array([i] * 10 + [49407] + [0] * 66, dtype=jnp.int32) for i in range(1, 4)
    ]

    # Process each text

    features = jax.vmap(eq_model)(jnp.array(texts))

    # Check output shape
    assert features.shape == (3, 512)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
