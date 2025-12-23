# eqx_clip_vit_b32.py
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from PIL import Image

from sarm.utils.tokenizer import load_tokenizer


def quick_gelu(x):
    # Matches CLIP QuickGELU: x * sigmoid(1.702 * x)
    return x * jax.nn.sigmoid(1.702 * x)


def build_causal_mask(context_length):
    """Build causal attention mask for text transformer."""
    mask = jnp.triu(jnp.ones((context_length, context_length)) * float("-inf"), k=1)
    return mask


def preprocess_image(image: Image.Image | np.ndarray) -> jax.Array:
    """
    Preprocess an image for CLIP (sarm/JAX version).

    Args:
        image: PIL Image

    Returns:
        Preprocessed image tensor (3, 224, 224)
    """
    # Resize to 224x224
    if isinstance(image, np.ndarray):
        C, H, W = image.shape
        image = (image * 255).astype(np.uint8)
        image = Image.fromarray(image.reshape(H, W, C))

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


def preprocess_images_batch(images: np.ndarray | jax.Array) -> jax.Array:
    """
    Vectorized preprocessing for a batch of images for CLIP.

    Args:
        images: Batch of images with shape (..., C, H, W) in range [0, 1]

    Returns:
        Preprocessed images with shape (..., C, 224, 224)
    """
    # Store original batch shape
    original_shape = images.shape
    batch_dims = original_shape[:-3]
    C, H, W = original_shape[-3:]

    # Flatten batch dimensions: (..., C, H, W) -> (N, C, H, W)
    images = images.reshape(-1, C, H, W)
    N = images.shape[0]

    # Transpose to (N, H, W, C) for jax.image.resize
    images = jnp.transpose(images, (0, 2, 3, 1))

    # Resize all images at once using jax
    images_resized = jax.image.resize(
        images, shape=(N, 224, 224, C), method="bicubic", antialias=True
    )

    # CLIP normalization constants
    mean = jnp.array([0.48145466, 0.4578275, 0.40821073])
    std = jnp.array([0.26862954, 0.26130258, 0.27577711])

    # Normalize: (N, 224, 224, C)
    images_normalized = (images_resized - mean) / std

    # Transpose back to (N, C, 224, 224)
    images_normalized = jnp.transpose(images_normalized, (0, 3, 1, 2))

    # Reshape back to original batch dimensions
    output_shape = batch_dims + (C, 224, 224)
    images_normalized = images_normalized.reshape(output_shape)

    return images_normalized


class MLP(eqx.Module):
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, d, mlp_ratio=4, key=jr.PRNGKey(0)):
        k1, k2 = jr.split(key)
        self.fc1 = eqx.nn.Linear(d, d * mlp_ratio, key=k1, use_bias=True)
        self.fc2 = eqx.nn.Linear(d * mlp_ratio, d, key=k2, use_bias=True)

    def __call__(self, x):
        x = jax.vmap(self.fc1)(x)
        x = quick_gelu(x)
        x = jax.vmap(self.fc2)(x)
        return x


class Attention(eqx.Module):
    """Unified attention with optional masking."""

    q: eqx.nn.Linear
    k: eqx.nn.Linear
    v: eqx.nn.Linear
    out: eqx.nn.Linear
    nheads: int
    head_dim: int
    scale: float

    def __init__(self, d, nheads, key=jr.PRNGKey(0)):
        kq, kk, kv, ko = jr.split(key, 4)
        self.q = eqx.nn.Linear(d, d, key=kq, use_bias=True)
        self.k = eqx.nn.Linear(d, d, key=kk, use_bias=True)
        self.v = eqx.nn.Linear(d, d, key=kv, use_bias=True)
        self.out = eqx.nn.Linear(d, d, key=ko, use_bias=True)
        self.nheads = nheads
        self.head_dim = d // nheads
        self.scale = 1.0 / jnp.sqrt(self.head_dim)

    def __call__(self, x, attn_mask=None):
        """
        x: (N, D) sequence
        attn_mask: optional (N, N) additive mask (e.g., causal mask for text)
        """
        N, D = x.shape

        def shape_heads(t):
            t = t.reshape(N, self.nheads, self.head_dim)
            return jnp.transpose(t, (1, 0, 2))  # (H, N, Hd)

        q = shape_heads(jax.vmap(self.q)(x))
        k = shape_heads(jax.vmap(self.k)(x))
        v = shape_heads(jax.vmap(self.v)(x))

        attn = jnp.einsum("hnd,hmd->hnm", q, k) * self.scale

        # Add mask if provided
        if attn_mask is not None:
            attn = attn + attn_mask

        attn = jax.nn.softmax(attn, axis=-1)
        out = jnp.einsum("hnm,hmd->hnd", attn, v)
        out = jnp.transpose(out, (1, 0, 2)).reshape(N, D)
        return jax.vmap(self.out)(out)


class Block(eqx.Module):
    """Transformer block for text with causal attention."""

    ln1: eqx.nn.LayerNorm
    ln2: eqx.nn.LayerNorm
    attn: Attention
    mlp: MLP

    def __init__(self, d, nheads, key=jr.PRNGKey(0)):
        ka, km = jr.split(key)
        self.ln1 = eqx.nn.LayerNorm(d, eps=1e-5)
        self.ln2 = eqx.nn.LayerNorm(d, eps=1e-5)
        self.attn = Attention(d, nheads, key=ka)
        self.mlp = MLP(d, mlp_ratio=4, key=km)

    def __call__(self, x, attn_mask=None):
        x = x + self.attn(jax.vmap(self.ln1)(x), attn_mask)
        x = x + self.mlp(jax.vmap(self.ln2)(x))
        return x


class ViTB32(eqx.Module):
    patch: eqx.nn.Conv2d
    cls: jax.Array
    pos: jax.Array
    blocks: list
    ln_pre: eqx.nn.LayerNorm
    ln_post: eqx.nn.LayerNorm
    proj: jax.Array  # (768, 512)

    image_size: int
    patch_size: int
    d: int
    nheads: int
    layers: int

    def __init__(
        self,
        image_size=224,
        patch_size=32,
        d=768,
        layers=12,
        nheads=12,
        key=jr.PRNGKey(0),
    ):
        k_conv, k_cls, k_pos, k_proj, k_blocks = jr.split(key, 5)
        self.patch = eqx.nn.Conv2d(
            3, d, kernel_size=patch_size, stride=patch_size, use_bias=False, key=k_conv
        )
        n = (image_size // patch_size) ** 2
        self.cls = jr.normal(k_cls, (1, d))
        self.pos = jr.normal(k_pos, (n + 1, d)) * 0.01
        self.blocks = [Block(d, nheads, key=jr.fold_in(k_blocks, i)) for i in range(layers)]
        self.ln_pre = eqx.nn.LayerNorm(d, eps=1e-5)
        self.ln_post = eqx.nn.LayerNorm(d, eps=1e-5)
        self.proj = jr.normal(k_proj, (d, 512)) * (d**-0.5)

        self.image_size = image_size
        self.patch_size = patch_size
        self.d = d
        self.nheads = nheads
        self.layers = layers

    def __call__(self, img):
        # img: (3, H, W), values already CLIP-normalized
        x = self.patch(img)  # (d, H/ps, W/ps)
        D, Hp, Wp = x.shape
        x = jnp.reshape(x, (D, Hp * Wp))
        x = jnp.transpose(x, (1, 0))  # (N, d)
        x = jnp.concatenate([self.cls, x], axis=0) + self.pos

        x = jax.vmap(self.ln_pre)(x)
        for blk in self.blocks:
            x = blk(x)
        x = jax.vmap(self.ln_post)(x)
        cls = x[0, :]  # (d,)
        feat = cls @ self.proj  # (512,)
        return feat


class TextTransformer(eqx.Module):
    """CLIP text encoder."""

    token_embedding: eqx.nn.Embedding
    positional_embedding: jax.Array
    blocks: list
    ln_final: eqx.nn.LayerNorm
    text_projection: jax.Array
    attn_mask: jax.Array

    context_length: int
    vocab_size: int
    d: int
    nheads: int
    layers: int

    def __init__(
        self,
        context_length=77,
        vocab_size=49408,
        d=512,
        layers=12,
        nheads=8,
        embed_dim=512,
        key=jr.PRNGKey(0),
    ):
        k_tok, k_pos, k_blocks, k_proj = jr.split(key, 4)

        self.token_embedding = eqx.nn.Embedding(vocab_size, d, key=k_tok)
        self.positional_embedding = jr.normal(k_pos, (context_length, d)) * 0.01
        self.blocks = [Block(d, nheads, key=jr.fold_in(k_blocks, i)) for i in range(layers)]
        self.ln_final = eqx.nn.LayerNorm(d, eps=1e-5)
        self.text_projection = jr.normal(k_proj, (d, embed_dim)) * (d**-0.5)
        self.attn_mask = build_causal_mask(context_length)

        self.context_length = context_length
        self.vocab_size = vocab_size
        self.d = d
        self.nheads = nheads
        self.layers = layers

    def __call__(self, text):
        """
        text: (N,) array of token indices
        Returns: (embed_dim,) text features
        """
        # Embed tokens
        x = jax.vmap(self.token_embedding)(text)  # (N, d)

        # Add positional embeddings
        seq_len = text.shape[0]
        x = x + self.positional_embedding[:seq_len]

        # Apply transformer blocks with causal masking
        mask = self.attn_mask[:seq_len, :seq_len]
        for blk in self.blocks:
            x = blk(x, mask)

        # Final layer norm
        x = jax.vmap(self.ln_final)(x)

        # Take features from the EOT embedding (highest index in sequence)
        eot_idx = jnp.argmax(text)
        x = x[eot_idx]

        # Project to embedding dimension
        if self.text_projection is not None:
            x = x @ self.text_projection

        return x


class CLIP(eqx.Module):
    """Complete CLIP model with vision and text encoders."""

    visual: ViTB32
    text: TextTransformer

    def __init__(
        self,
        # Vision params
        image_size=224,
        patch_size=32,
        vision_width=768,
        vision_layers=12,
        vision_heads=12,
        # Text params
        context_length=77,
        vocab_size=49408,
        text_width=512,
        text_layers=12,
        text_heads=8,
        # Shared
        embed_dim=512,
        key=jr.PRNGKey(0),
    ):
        k_vis, k_text = jr.split(key)

        self.visual = ViTB32(
            image_size=image_size,
            patch_size=patch_size,
            d=vision_width,
            layers=vision_layers,
            nheads=vision_heads,
            key=k_vis,
        )

        self.text = TextTransformer(
            context_length=context_length,
            vocab_size=vocab_size,
            d=text_width,
            layers=text_layers,
            nheads=text_heads,
            embed_dim=embed_dim,
            key=k_text,
        )

    def encode_image(self, image):
        """Encode image to features."""
        return self.visual(image)

    def encode_text(self, text):
        """Encode text to features."""
        return self.text(text)

    def __call__(self, image, text):
        """Encode both image and text, return normalized features."""
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # Normalize features
        image_features = image_features / jnp.linalg.norm(image_features)
        text_features = text_features / jnp.linalg.norm(text_features)

        return image_features, text_features


def load_vision_npz(model: ViTB32, path: str) -> ViTB32:
    """Load vision weights from npz file."""
    data = np.load(path)
    # Patch conv
    model = eqx.tree_at(lambda m: m.patch.weight, model, jnp.asarray(data["visual.patch.weight"]))

    # Tokens/pos
    model = eqx.tree_at(lambda m: m.cls, model, jnp.asarray(data["visual.cls"]))
    model = eqx.tree_at(lambda m: m.pos, model, jnp.asarray(data["visual.pos"]))

    # Blocks
    def assign_block(b: Block, i: int):
        b = eqx.tree_at(
            lambda x: x.ln1.weight,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.ln1.weight"]),
        )
        b = eqx.tree_at(lambda x: x.ln1.bias, b, jnp.asarray(data[f"visual.blocks.{i}.ln1.bias"]))
        b = eqx.tree_at(
            lambda x: x.ln2.weight,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.ln2.weight"]),
        )
        b = eqx.tree_at(lambda x: x.ln2.bias, b, jnp.asarray(data[f"visual.blocks.{i}.ln2.bias"]))

        b = eqx.tree_at(
            lambda x: x.attn.q.weight,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.attn.q.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.q.bias,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.attn.q.bias"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.k.weight,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.attn.k.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.k.bias,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.attn.k.bias"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.v.weight,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.attn.v.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.v.bias,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.attn.v.bias"]),
        )

        b = eqx.tree_at(
            lambda x: x.attn.out.weight,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.attn.out.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.out.bias,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.attn.out.bias"]),
        )

        b = eqx.tree_at(
            lambda x: x.mlp.fc1.weight,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.mlp.fc1.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc1.bias,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.mlp.fc1.bias"]),
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc2.weight,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.mlp.fc2.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc2.bias,
            b,
            jnp.asarray(data[f"visual.blocks.{i}.mlp.fc2.bias"]),
        )
        return b

    blocks = [assign_block(b, i) for i, b in enumerate(model.blocks)]
    model = eqx.tree_at(lambda m: m.blocks, model, blocks)

    # Final LN
    model = eqx.tree_at(lambda m: m.ln_pre.weight, model, jnp.asarray(data["visual.ln_pre.weight"]))
    model = eqx.tree_at(lambda m: m.ln_pre.bias, model, jnp.asarray(data["visual.ln_pre.bias"]))
    model = eqx.tree_at(
        lambda m: m.ln_post.weight, model, jnp.asarray(data["visual.ln_post.weight"])
    )
    model = eqx.tree_at(lambda m: m.ln_post.bias, model, jnp.asarray(data["visual.ln_post.bias"]))

    # Projection (PyTorch stored as [768,512], Equinox uses the same matmul ordering cls @ proj)
    model = eqx.tree_at(lambda m: m.proj, model, jnp.asarray(data["visual.proj.weight"]))

    return model


def load_text_npz(model: TextTransformer, path: str) -> TextTransformer:
    """Load text weights from npz file."""
    data = np.load(path)

    # Token and positional embeddings
    model = eqx.tree_at(
        lambda m: m.token_embedding.weight,
        model,
        jnp.asarray(data["text.token_embedding.weight"]),
    )
    model = eqx.tree_at(
        lambda m: m.positional_embedding,
        model,
        jnp.asarray(data["text.positional_embedding"]),
    )

    # Transformer blocks
    def assign_text_block(b: Block, i: int):
        b = eqx.tree_at(lambda x: x.ln1.weight, b, jnp.asarray(data[f"text.blocks.{i}.ln1.weight"]))
        b = eqx.tree_at(lambda x: x.ln1.bias, b, jnp.asarray(data[f"text.blocks.{i}.ln1.bias"]))
        b = eqx.tree_at(lambda x: x.ln2.weight, b, jnp.asarray(data[f"text.blocks.{i}.ln2.weight"]))
        b = eqx.tree_at(lambda x: x.ln2.bias, b, jnp.asarray(data[f"text.blocks.{i}.ln2.bias"]))

        b = eqx.tree_at(
            lambda x: x.attn.q.weight,
            b,
            jnp.asarray(data[f"text.blocks.{i}.attn.q.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.q.bias,
            b,
            jnp.asarray(data[f"text.blocks.{i}.attn.q.bias"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.k.weight,
            b,
            jnp.asarray(data[f"text.blocks.{i}.attn.k.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.k.bias,
            b,
            jnp.asarray(data[f"text.blocks.{i}.attn.k.bias"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.v.weight,
            b,
            jnp.asarray(data[f"text.blocks.{i}.attn.v.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.v.bias,
            b,
            jnp.asarray(data[f"text.blocks.{i}.attn.v.bias"]),
        )

        b = eqx.tree_at(
            lambda x: x.attn.out.weight,
            b,
            jnp.asarray(data[f"text.blocks.{i}.attn.out.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.attn.out.bias,
            b,
            jnp.asarray(data[f"text.blocks.{i}.attn.out.bias"]),
        )

        b = eqx.tree_at(
            lambda x: x.mlp.fc1.weight,
            b,
            jnp.asarray(data[f"text.blocks.{i}.mlp.fc1.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc1.bias,
            b,
            jnp.asarray(data[f"text.blocks.{i}.mlp.fc1.bias"]),
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc2.weight,
            b,
            jnp.asarray(data[f"text.blocks.{i}.mlp.fc2.weight"]),
        )
        b = eqx.tree_at(
            lambda x: x.mlp.fc2.bias,
            b,
            jnp.asarray(data[f"text.blocks.{i}.mlp.fc2.bias"]),
        )
        return b

    blocks = [assign_text_block(b, i) for i, b in enumerate(model.blocks)]
    model = eqx.tree_at(lambda m: m.blocks, model, blocks)

    # Final layer norm
    model = eqx.tree_at(
        lambda m: m.ln_final.weight, model, jnp.asarray(data["text.ln_final.weight"])
    )
    model = eqx.tree_at(lambda m: m.ln_final.bias, model, jnp.asarray(data["text.ln_final.bias"]))

    # Text projection
    model = eqx.tree_at(
        lambda m: m.text_projection, model, jnp.asarray(data["text.text_projection"])
    )

    return model


def load_clip_npz(model: CLIP, path: str) -> CLIP:
    """Load complete CLIP model from npz file."""
    model = eqx.tree_at(lambda m: m.visual, model, load_vision_npz(model.visual, path))
    model = eqx.tree_at(lambda m: m.text, model, load_text_npz(model.text, path))
    return model
