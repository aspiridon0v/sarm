# tests/test_clip_vit_b32_equivalence.py
import os

import jax
import jax.numpy as jnp
import numpy as np
import open_clip
import pytest
import torch
from PIL import Image, ImageDraw

from sarm.model.clip import ViTB32, load_vision_npz
from sarm.utils.convert_clip import main as export_clip_weights


@pytest.fixture(scope="session")
def pt_model_and_preprocess(torch_device):
    # Load PyTorch CLIP ViT-B/32 (openai weights) and its official preprocess
    # Force quick_gelu=True to match the original OpenAI training
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", force_quick_gelu=True
    )
    model.eval()
    model.to(torch_device)
    return model, preprocess, torch_device


def _make_test_images():
    """Create two deterministic 224x224 RGB images as PIL Images."""
    # 1) Solid color with a small contrasting square
    img1 = Image.new("RGB", (224, 224), (220, 30, 30))
    d = ImageDraw.Draw(img1)
    d.rectangle([64, 64, 160, 160], fill=(30, 220, 30))

    # 2) Horizontal gradient
    img2 = Image.new("RGB", (224, 224))
    pixels = img2.load()
    for x in range(224):
        val = int(255 * x / 223)
        for y in range(224):
            pixels[x, y] = (val, 255 - val, (val // 2))
    return [img1, img2]


def _torch_forward_visual(model, imgs_pt):
    with torch.no_grad():
        out = model.visual(imgs_pt)  # Expect (B, 512)
    if (
        not isinstance(out, torch.Tensor)
        or out.ndim != 2
        or out.shape[-1] != model.visual.output_dim
    ):
        raise RuntimeError(
            f"Unexpected visual forward output shape: {getattr(out, 'shape', None)}"
        )
    return out


def test_vit_b32_image_features_match_pytorch(pt_model_and_preprocess, ensure_weights):
    model, preprocess, torch_device = pt_model_and_preprocess

    # ----- Build (and load weights into) the Equinox model -----
    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_vision_npz(eq_model, ensure_weights)
    eq_model = jax.vmap(eq_model)

    # ----- Make two test images and preprocess with official transforms -----
    pil_imgs = _make_test_images()
    imgs_pt = torch.stack([preprocess(im.convert("RGB")) for im in pil_imgs], dim=0).to(
        torch_device
    )  # (B,3,224,224)

    # ----- PyTorch path -----
    pt_feat = _torch_forward_visual(model, imgs_pt).cpu().numpy()  # (B,512)

    # ----- Equinox path -----
    imgs_np = imgs_pt.cpu().numpy()  # (B,3,224,224) already CLIP-normalized
    imgs_jax = jnp.asarray(imgs_np)
    eq_feat = np.array(eq_model(imgs_jax))  # (B,512)

    # ----- Compare -----
    assert pt_feat.shape == eq_feat.shape == (len(pil_imgs), 512)

    # Tight tolerances; relax very slightly if different BLAS/backends cause tiny drift
    # Using atol=2e-5 to account for accumulated floating point differences between PyTorch and JAX
    atol = 2e-5
    rtol = 1e-5

    # Calculate differences for better error messages
    max_abs = np.max(np.abs(eq_feat - pt_feat))
    mean_abs = np.mean(np.abs(eq_feat - pt_feat))

    assert np.allclose(
        eq_feat, pt_feat, atol=atol, rtol=rtol
    ), f"Equinox features differ from PyTorch (max|diff|={max_abs:.3e}, mean|diff|={mean_abs:.3e})"


def test_weight_loading(pt_model_and_preprocess, ensure_weights):
    """Test that weights are loaded correctly from PyTorch to Equinox."""
    model, _, torch_device = pt_model_and_preprocess
    pt_visual = model.visual

    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_vision_npz(eq_model, ensure_weights)

    # Test MLP weights in first block
    pt_fc1_weight = (
        pt_visual.transformer.resblocks[0].mlp.c_fc.weight.detach().cpu().numpy()
    )
    pt_fc1_bias = (
        pt_visual.transformer.resblocks[0].mlp.c_fc.bias.detach().cpu().numpy()
    )
    eq_fc1_weight = np.array(eq_model.blocks[0].mlp.fc1.weight)
    eq_fc1_bias = np.array(eq_model.blocks[0].mlp.fc1.bias)

    # Weights should be exactly identical
    assert np.allclose(
        pt_fc1_weight, eq_fc1_weight, atol=0, rtol=0
    ), "fc1 weights don't match"
    assert np.allclose(
        pt_fc1_bias, eq_fc1_bias, atol=0, rtol=0
    ), "fc1 bias doesn't match"


def test_linear_layer_equivalence(pt_model_and_preprocess, ensure_weights):
    """Test that Equinox Linear layers produce the same output as PyTorch."""
    model, _, torch_device = pt_model_and_preprocess
    pt_visual = model.visual

    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_vision_npz(eq_model, ensure_weights)

    # Create test input
    test_input_np = np.random.randn(50, 768).astype(np.float32)
    test_input_pt = torch.from_numpy(test_input_np).to(torch_device)
    test_input_jax = jnp.array(test_input_np)

    # Test PyTorch
    with torch.no_grad():
        pt_output = (
            pt_visual.transformer.resblocks[0].mlp.c_fc(test_input_pt).cpu().numpy()
        )

    # Test Equinox
    eq_output = np.array(jax.vmap(eq_model.blocks[0].mlp.fc1)(test_input_jax))

    # Should match to numerical precision
    assert np.allclose(
        pt_output, eq_output, atol=1e-5, rtol=1e-5
    ), f"Mean absolute difference: {np.mean(np.abs(pt_output - eq_output))}, Max absolute difference: {np.max(np.abs(pt_output - eq_output))}"


def test_patch_embeddings(pt_model_and_preprocess, ensure_weights):
    """Test that patch embeddings (conv layer) match between PyTorch and Equinox."""
    model, preprocess, torch_device = pt_model_and_preprocess
    pt_visual = model.visual

    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_vision_npz(eq_model, ensure_weights)

    # Create a test image
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    img_pt = preprocess(img).unsqueeze(0).to(torch_device)
    img_jax = jnp.array(img_pt.cpu().numpy())

    # PyTorch patch embedding
    with torch.no_grad():
        pt_patches = pt_visual.conv1(img_pt).cpu().numpy()  # (1, 768, 7, 7)

    # Equinox patch embedding
    eq_patches = np.array(eq_model.patch(img_jax[0]))  # (768, 7, 7)

    # Should match very closely (minor floating point differences expected)
    assert np.allclose(
        pt_patches[0], eq_patches, atol=1e-5, rtol=1e-5
    ), f"Mean absolute difference: {np.mean(np.abs(pt_patches[0] - eq_patches))}, Max absolute difference: {np.max(np.abs(pt_patches[0] - eq_patches))}"


def test_layer_norm_equivalence(pt_model_and_preprocess, ensure_weights):
    """Test that LayerNorm produces the same output."""
    model, _, torch_device = pt_model_and_preprocess
    pt_visual = model.visual

    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_vision_npz(eq_model, ensure_weights)

    # Create test input
    test_input_np = np.random.randn(50, 768).astype(np.float32)
    test_input_pt = torch.from_numpy(test_input_np).to(torch_device)
    test_input_jax = jnp.array(test_input_np)

    # PyTorch LayerNorm
    with torch.no_grad():
        pt_output = pt_visual.ln_pre(test_input_pt[None, :, :])[0].cpu().numpy()

    # Equinox LayerNorm
    eq_output = np.array(jax.vmap(eq_model.ln_pre)(test_input_jax))

    # Should match to numerical precision
    assert np.allclose(
        pt_output, eq_output, atol=1e-5, rtol=1e-5
    ), f"Mean absolute difference: {np.mean(np.abs(pt_output - eq_output))}, Max absolute difference: {np.max(np.abs(pt_output - eq_output))}"


def test_attention_mechanism(pt_model_and_preprocess, ensure_weights):
    """Test that the attention mechanism produces matching outputs."""
    model, preprocess, torch_device = pt_model_and_preprocess
    pt_visual = model.visual

    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_vision_npz(eq_model, ensure_weights)

    # Create test input matching the output of ln_pre
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    img_pt = preprocess(img).unsqueeze(0).to(torch_device)
    img_jax = jnp.array(img_pt.cpu().numpy())

    # Get to the input of first attention block
    with torch.no_grad():
        x_pt = pt_visual.conv1(img_pt)
        x_pt = x_pt.reshape(x_pt.shape[0], x_pt.shape[1], -1).permute(0, 2, 1)
        x_pt = torch.cat(
            [pt_visual.class_embedding.to(x_pt.dtype)[None, None, :], x_pt], dim=1
        )
        x_pt = x_pt + pt_visual.positional_embedding.to(x_pt.dtype)[None, :, :]
        x_pt = pt_visual.ln_pre(x_pt)[0]  # (50, 768)
        x_pt_ln1 = pt_visual.transformer.resblocks[0].ln_1(x_pt)

    x_eq = eq_model.patch(img_jax[0])
    D, Hp, Wp = x_eq.shape
    x_eq = jnp.reshape(x_eq, (D, Hp * Wp))
    x_eq = jnp.transpose(x_eq, (1, 0))
    x_eq = jnp.concatenate([eq_model.cls, x_eq], axis=0) + eq_model.pos
    x_eq = jax.vmap(eq_model.ln_pre)(x_eq)
    x_eq_ln1 = jax.vmap(eq_model.blocks[0].ln1)(x_eq)

    # Test attention
    with torch.no_grad():
        pt_attn_out = (
            pt_visual.transformer.resblocks[0]
            .attn(
                x_pt_ln1[None, :, :],
                x_pt_ln1[None, :, :],
                x_pt_ln1[None, :, :],
                need_weights=False,
            )[0][0]
            .cpu()
            .numpy()
        )

    eq_attn_out = np.array(eq_model.blocks[0].attn(x_eq_ln1))

    # Attention should match very closely
    max_diff = np.max(np.abs(pt_attn_out - eq_attn_out))
    assert (
        max_diff < 1e-5
    ), f"Attention outputs differ by {max_diff}. Mean absolute difference: {np.mean(np.abs(pt_attn_out - eq_attn_out))}, Max absolute difference: {np.max(np.abs(pt_attn_out - eq_attn_out))}"


def test_mlp_with_quick_gelu(pt_model_and_preprocess, ensure_weights):
    """Test that MLP with QuickGELU produces matching outputs."""
    model, _, torch_device = pt_model_and_preprocess
    pt_visual = model.visual

    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_vision_npz(eq_model, ensure_weights)

    # Create test input
    test_input_np = np.random.randn(50, 768).astype(np.float32)
    test_input_pt = torch.from_numpy(test_input_np).to(torch_device)
    test_input_jax = jnp.array(test_input_np)

    # PyTorch MLP (should be using QuickGELU now)
    with torch.no_grad():
        pt_output = pt_visual.transformer.resblocks[0].mlp(test_input_pt).cpu().numpy()

    # Equinox MLP
    eq_output = np.array(eq_model.blocks[0].mlp(test_input_jax))

    # Should match closely (QuickGELU is deterministic)
    max_diff = np.max(np.abs(pt_output - eq_output))
    assert (
        max_diff < 1e-4
    ), f"MLP outputs differ by {max_diff}. Mean absolute difference: {np.mean(np.abs(pt_output - eq_output))}, Max absolute difference: {np.max(np.abs(pt_output - eq_output))}"


def test_full_transformer_block(pt_model_and_preprocess, ensure_weights):
    """Test that a complete transformer block produces matching outputs."""
    model, preprocess, torch_device = pt_model_and_preprocess
    pt_visual = model.visual

    eq_model = ViTB32(image_size=224, patch_size=32, d=768, layers=12, nheads=12)
    eq_model = load_vision_npz(eq_model, ensure_weights)

    # Get to block 0 input
    img = Image.new("RGB", (224, 224), (128, 128, 128))
    img_pt = preprocess(img).unsqueeze(0).to(torch_device)
    img_jax = jnp.array(img_pt.cpu().numpy())

    with torch.no_grad():
        x_pt = pt_visual.conv1(img_pt)
        x_pt = x_pt.reshape(x_pt.shape[0], x_pt.shape[1], -1).permute(0, 2, 1)
        x_pt = torch.cat(
            [pt_visual.class_embedding.to(x_pt.dtype)[None, None, :], x_pt], dim=1
        )
        x_pt = x_pt + pt_visual.positional_embedding.to(x_pt.dtype)[None, :, :]
        x_pt = pt_visual.ln_pre(x_pt)[0]

    x_eq = eq_model.patch(img_jax[0])
    D, Hp, Wp = x_eq.shape
    x_eq = jnp.reshape(x_eq, (D, Hp * Wp))
    x_eq = jnp.transpose(x_eq, (1, 0))
    x_eq = jnp.concatenate([eq_model.cls, x_eq], axis=0) + eq_model.pos
    x_eq = jax.vmap(eq_model.ln_pre)(x_eq)

    # Run through first block
    with torch.no_grad():
        pt_block_out = (
            pt_visual.transformer.resblocks[0](x_pt[None, :, :])[0].cpu().numpy()
        )

    eq_block_out = np.array(eq_model.blocks[0](x_eq))

    # Full block should still match well
    max_diff = np.max(np.abs(pt_block_out - eq_block_out))
    mean_diff = np.mean(np.abs(pt_block_out - eq_block_out))
    assert max_diff < 5e-5, f"Block outputs differ by max={max_diff}, mean={mean_diff}"
