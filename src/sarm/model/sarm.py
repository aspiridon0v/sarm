import datetime
import logging
from typing import Callable

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
from jaxtyping import PyTree, PRNGKeyArray, Array

from sarm.model.clip import CLIP, Block, load_clip_npz

logger = logging.getLogger(__name__)


@eqx.filter_jit
def clip_inference(
        clip_model: CLIP,
        images: jax.Array,
        text_tokens: jax.Array,
        img_chunk_size: int = 64,  # Adjust based on your GPU VRAM (64-128 is usually safe)
        text_chunk_size: int = 512,
):
    """
    Memory-efficient CLIP inference using internal micro-batching.
    """
    B, N, T, C, H, W = images.shape

    # --- Image Processing ---
    # Flatten: (Total_Images, C, H, W)
    images_reshaped = images.reshape((B * N * T, C, H, W))

    # Apply model in chunks
    img_features = _batched_forward(
        lambda x: clip_model.encode_image_batch(x),  # Function to apply to each chunk
        images_reshaped,
        img_chunk_size
    )

    # Reshape back: (B, N, T, d_vis)
    img_features = img_features.reshape((B, N, T, -1))

    # --- Text Processing ---
    # Flatten: (Total_Texts, max_len)
    text_tokens_reshaped = text_tokens.reshape((B * T, -1))

    # Apply model in chunks (Text is lighter, can use larger chunks)
    text_features = _batched_forward(
        lambda x: clip_model.encode_text_batch(x),
        text_tokens_reshaped,
        text_chunk_size
    )

    # Reshape back: (B, T, d_text)
    text_features = text_features.reshape((B, T, -1))

    # Stop gradients to save memory and ensure safety
    return jax.lax.stop_gradient(img_features), jax.lax.stop_gradient(text_features)

def _batched_forward(apply_fn: Callable, inputs: jax.Array, batch_size: int) -> jax.Array:
    """Chunk-processes a large tensor using jax.lax.map.

    Keeps peak memory low by only materializing activations for 'batch_size' items.
    """
    n = inputs.shape[0]

    # Calculate padding to ensure shape is divisible by batch_size
    num_chunks = (n + batch_size - 1) // batch_size
    pad_amt = num_chunks * batch_size - n

    # Pad the inputs
    pad_width = [(0, pad_amt)] + [(0, 0)] * (inputs.ndim - 1)
    inputs_padded = jnp.pad(inputs, pad_width, mode='edge')

    # Reshape into (Num_Chunks, Batch_Size, ...)
    input_chunks = inputs_padded.reshape((num_chunks, batch_size, *inputs.shape[1:]))

    # Run the model sequentially over chunks
    output_chunks = jax.lax.map(apply_fn, input_chunks)

    # Reshape back and remove padding
    output_flattened = output_chunks.reshape((-1, output_chunks.shape[-1]))

    return output_flattened[:n]


class ProgressTransformer(eqx.Module):

    vis_proj: eqx.nn.Linear
    text_proj: eqx.nn.Linear
    state_proj: eqx.nn.Linear
    final_proj: dict
    fusion_mlp: eqx.nn.Sequential
    blocks: list
    positional_embedding: jax.Array

    def __init__(
        self,
        d_model: int = 512,
        nheads: int = 8,
        layers: int = 12,
        vis_embed_dim: int = 512,
        text_embed_dim: int = 512,
        state_dim: int = 14,
        num_cameras: int = 1,
        dropout: float = 0.0,
        key=None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        k_blocks, k_vis, k_text, k_state, k_fusion, k_sparse, k_dense = jr.split(key, 7)
        self.vis_proj = eqx.nn.Linear(vis_embed_dim, d_model, key=k_vis)
        self.text_proj = eqx.nn.Linear(text_embed_dim, d_model, key=k_text)
        self.state_proj = eqx.nn.Linear(state_dim, d_model, key=k_state)
        self.final_proj = {
            "sparse": eqx.nn.Linear(d_model, 1, key=k_sparse),
            "dense": eqx.nn.Linear(d_model, 1, key=k_dense),
        }
        self.fusion_mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(
                    (num_cameras + 3) * d_model,
                ),
                eqx.nn.Linear((num_cameras + 3) * d_model, d_model, key=k_fusion),
                eqx.nn.Lambda(jax.nn.relu),
            ],
        )

        self.blocks = [Block(d_model, nheads, key=jr.fold_in(k_blocks, i), dropout=dropout) for i in range(layers)]
        self.positional_embedding = jnp.zeros((1, d_model))

    def _subtask_encoding(self, subtask: jax.Array, d_model: int):
        """Encode subtask features.


        Args:
            subtask (jax.Array): Subtask features of shape (T, C)

        Returns:
            jax.Array: Subtask features of shape (T, d_model)
        """
        if subtask.shape[-1] == d_model:
            return subtask
        elif subtask.shape[-1] > d_model:
            return subtask[:, :d_model]
        else:
            return jnp.concatenate(
                [
                    subtask,
                    jnp.zeros(
                        (
                            subtask.shape[0],
                            d_model - subtask.shape[-1],
                        )
                    ),
                ],
                axis=-1,
            )

    def _build_mask(self, timesteps: int, length: int, num_cameras: int):
        """Build mask for the subtask transformer.

        Args:
            length (int): Length of the sequence

        Returns:
            jax.Array: Mask of shape ((N+3)*T, (N+3)*T)
        """
        mask_1d = jnp.arange(timesteps) < length
        mask_1d = jnp.where(mask_1d, 0.0, float("-inf"))
        mask_1d = einops.repeat(mask_1d, "t -> (n t)", n=num_cameras + 3)
        # Only mask keys (columns), not queries (rows), to prevent all-inf rows
        mask = mask_1d[None, :]  # (1, (N+3)*T) -> broadcasts over query dimension
        return mask

    def __call__(
        self,
        img_features: jax.Array,
        text_features: jax.Array,
        state: jax.Array,
        subtask: jax.Array,
        length: int,
        dense_schema: jax.Array,
        key: PRNGKeyArray | None = None
    ):
        """Forward pass for the subtask transformer.

        Args:
            img_features (jax.Array): Image features of shape (N, T, d_vis)
            text_features (jax.Array): Text features of shape (T, d_text)
            state (jax.Array): State features of shape (T, d_state)
            subtask (jax.Array): Subtask features of shape (T, C)
            length (jax.Array): Length of the sequence
            dense_schema (jax.Array): Boolean if the schema is dense
        Returns:
            jax.Array: Output features of shape (T)
        """
        N, T, D = img_features.shape
        d_model = int(self.vis_proj.out_features)
        img_features = jax.vmap(jax.vmap(self.vis_proj))(img_features)  # (N, T, d_model)
        text_features = jax.vmap(self.text_proj)(text_features)[None, ...]  # (1, T, d_model)
        state_features = jax.vmap(self.state_proj)(state)[None, ...]  # (1, T, d_model)
        subtask_features = self._subtask_encoding(subtask, d_model)[None, ...]  # (1, T, d_model)

        # Combine features
        features = jnp.concatenate(
            [img_features, text_features, state_features, subtask_features], axis=0
        )  # (N + 3, T, d_model)

        features = features.at[:N, 0, :].add(self.positional_embedding)
        features = einops.rearrange(features, "n t d -> (n t) d")  # ((N+3)*T, d_model)

        mask = self._build_mask(T, length, N)  # ((N+3)*T, (N+3)*T)

        # Apply transformer blocks
        keys = jr.split(key, len(self.blocks)) if key is not None else [None]*len(self.blocks)
        for n, block in enumerate(self.blocks):
            features = block(features, mask, key=keys[n])

        features = einops.rearrange(features, "(n t) d -> t (n d)", n=N + 3, t=T)
        features = jax.vmap(self.fusion_mlp)(features)  # (T, d_model)

        features = jax.vmap(self.final_proj["dense"])(features).squeeze(
            -1
        )  # (T,) TODO: add conditional sparse projection

        return jax.vmap(jax.nn.sigmoid)(features)  # (T,)

    def save_checkpoint(self, path: str):
        eqx.tree_serialise_leaves(path, self)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        model = eqx.tree_deserialise_leaves(path, self)
        logger.info(f"Loaded checkpoint from {path}")
        return model


class StageTransformer(eqx.Module):

    vis_proj: eqx.nn.Linear
    text_proj: eqx.nn.Linear
    state_proj: eqx.nn.Linear
    final_proj: dict
    fusion_mlp: eqx.nn.Sequential
    blocks: list
    positional_embedding: jax.Array

    def __init__(
        self,
        d_model: int = 512,
        nheads: int = 8,
        layers: int = 12,
        vis_embed_dim: int = 512,
        text_embed_dim: int = 512,
        state_dim: int = 14,
        num_cameras: int = 1,
        num_classes_sparse: int = 4,
        num_classes_dense: int = 8,
        dropout: float = 0.0,
        key=None,
    ):
        if key is None:
            key = jr.PRNGKey(0)
        k_blocks, k_vis, k_text, k_state, k_fusion, k_sparse, k_dense = jr.split(key, 7)
        self.vis_proj = eqx.nn.Linear(vis_embed_dim, d_model, key=k_vis)
        self.text_proj = eqx.nn.Linear(text_embed_dim, d_model, key=k_text)
        self.state_proj = eqx.nn.Linear(state_dim, d_model, key=k_state)
        self.final_proj = {
            "sparse": eqx.nn.Linear(d_model, num_classes_sparse, key=k_sparse),
            "dense": eqx.nn.Linear(d_model, num_classes_dense, key=k_dense),
        }
        self.fusion_mlp = eqx.nn.Sequential(
            [
                eqx.nn.LayerNorm(
                    (num_cameras + 2) * d_model,
                ),
                eqx.nn.Linear((num_cameras + 2) * d_model, d_model, key=k_fusion),
                eqx.nn.Lambda(jax.nn.relu),
            ],
        )

        self.blocks = [Block(d_model, nheads, key=jr.fold_in(k_blocks, i), dropout=dropout) for i in range(layers)]
        self.positional_embedding = jnp.zeros((1, d_model))

    def _build_mask(self, timesteps: int, length: int, num_cameras: int):
        """Build mask for the subtask transformer.

        Args:
            length (int): Length of the sequence

        Returns:
            jax.Array: Mask of shape ((N+2)*T, (N+2)*T)
        """
        mask_1d = jnp.arange(timesteps) < length
        mask_1d = jnp.where(mask_1d, 0.0, float("-inf"))
        mask_1d = einops.repeat(mask_1d, "t -> (n t)", n=num_cameras + 2)
        # Only mask keys (columns), not queries (rows), to prevent all-inf rows
        mask = mask_1d[None, :]  # (1, (N+2)*T) -> broadcasts over query dimension
        return mask

    def __call__(
        self,
        img_features: jax.Array,
        text_features: jax.Array,
        state: jax.Array,
        length: int,
        dense_schema: jax.Array,
        key: PRNGKeyArray | None = None
    ):
        """Forward pass for the stage transformer.

        Args:
            img_features (jax.Array): Image features of shape (N, T, d_vis)
            text_features (jax.Array): Text features of shape (T, d_text)
            state (jax.Array): State features of shape (T, d_state)
            length (int): Length of the sequence
            dense_schema (jax.Array): Boolean if the schema is dense
        Returns:
            jax.Array: Output features of shape (T)
        """
        N, T, d_vis = img_features.shape
        img_features = jax.vmap(jax.vmap(self.vis_proj))(img_features)  # (N, T, d_model)
        text_features = jax.vmap(self.text_proj)(text_features)[None, ...]  # (1, T, d_model)
        state_features = jax.vmap(self.state_proj)(state)[None, ...]  # (1, T, d_model)

        # Combine features
        features = jnp.concatenate(
            [img_features, text_features, state_features], axis=0
        )  # (N + 2, T, d_model)

        features = features.at[:N, 0, :].add(self.positional_embedding)
        features = einops.rearrange(features, "n t d -> (n t) d")  # ((N+2)*T, d_model)

        mask = self._build_mask(T, length, N)  # ((N+2)*T, (N+2)*T)

        # Apply transformer blocks
        # Apply transformer blocks
        keys = jr.split(key, len(self.blocks)) if key is not None else [None]*len(self.blocks)
        for n, block in enumerate(self.blocks):
            features = block(features, mask, key=keys[n])

        features = einops.rearrange(features, "(n t) d -> t (n d)", n=N + 2, t=T)
        features = jax.vmap(self.fusion_mlp)(features)  # (T, d_model)

        logits = jax.vmap(self.final_proj["dense"])(
            features
        )  # (T, C) TODO: add conditional sparse projection

        return logits  # (T, C)

    def save_checkpoint(self, path: str):
        eqx.tree_serialise_leaves(path, self)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> PyTree:
        model = eqx.tree_deserialise_leaves(path, self)
        logger.info(f"Loaded checkpoint from {path}")
        return model


class Sarm(eqx.Module):

    progress_transformer: ProgressTransformer
    stage_transformer: StageTransformer
    clip_model: CLIP

    def __init__(
        self,
        progress_transformer: ProgressTransformer,
        stage_transformer: StageTransformer,
        clip_model: CLIP,
    ):
        self.progress_transformer = progress_transformer
        self.stage_transformer = stage_transformer
        self.clip_model = clip_model

    def clip_inference(
        self,
        images: jax.Array,
        text_tokens: jax.Array,
        img_chunk_size: int = 64,
        text_chunk_size: int = 512,
    ) -> tuple[jax.Array, jax.Array]:
        return clip_inference(self.clip_model,
                              images,
                              text_tokens,
                              img_chunk_size,
                              text_chunk_size)

    def predict_stage(
        self,
        img_features: jax.Array,
        text_features: jax.Array,
        state: jax.Array,
        length: jax.Array,
        dense_schema: bool,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        """Predict stage logits.

        Args:
            img_features: Image features of shape (B, N, T, d_vis)
            text_features: Text features of shape (B, T, d_text)
            state: State features of shape (B, T, d_state)
            length: Sequence lengths of shape (B,)
            dense_schema: Whether using dense annotation schema
            key: Optional PRNG key for dropout

        Returns:
            Stage logits of shape (B, T, num_classes)
        """
        B = img_features.shape[0]
        if key is None:
            return jax.vmap(
                self.stage_transformer,
                in_axes=(0, 0, 0, 0, None, None),
            )(img_features, text_features, state, length, dense_schema, None)
        keys = jr.split(key, B)
        return jax.vmap(
            self.stage_transformer,
            in_axes=(0, 0, 0, 0, None, 0),
        )(img_features, text_features, state, length, dense_schema, keys)

    def predict_progress(
        self,
        img_features: jax.Array,
        text_features: jax.Array,
        state: jax.Array,
        stage_emb: jax.Array,
        length: jax.Array,
        dense_schema: bool,
        key: PRNGKeyArray | None = None,
    ) -> jax.Array:
        """Predict progress within current stage.

        Args:
            img_features: Image features of shape (B, N, T, d_vis)
            text_features: Text features of shape (B, T, d_text)
            state: State features of shape (B, T, d_state)
            stage_emb: One-hot stage embedding of shape (B, T, num_classes)
            length: Sequence lengths of shape (B,)
            dense_schema: Whether using dense annotation schema
            key: Optional PRNG key for dropout

        Returns:
            Progress predictions of shape (B, T) in range [0, 1]
        """
        B = img_features.shape[0]
        if key is None:
            return jax.vmap(
                self.progress_transformer,
                in_axes=(0, 0, 0, 0, 0, None, None),
            )(img_features, text_features, state, stage_emb, length, dense_schema, None)
        keys = jr.split(key, B)
        return jax.vmap(
            self.progress_transformer,
            in_axes=(0, 0, 0, 0, 0, None, 0),
        )(img_features, text_features, state, stage_emb, length, dense_schema, keys)

    def __call__(
        self,
        images: jax.Array,
        text_tokens: jax.Array,
        state: jax.Array,
        length: jax.Array,
        dense_schema: bool,
        img_chunk_size: int = 64,
        key: PRNGKeyArray | None = None,
    ) -> dict[str, jax.Array]:
        """Full forward pass: encode features, predict stage, then predict progress.

        Args:
            images: Image tensor of shape (B, N, T, C, H, W)
            text_tokens: Text tokens of shape (B, T, seq_len)
            state: State features of shape (B, T, d_state)
            length: Sequence lengths of shape (B,)
            dense_schema: Whether using dense annotation schema
            img_chunk_size: Batch size for CLIP image encoding
            key: Optional PRNG key for dropout

        Returns:
            Dictionary with keys:
                - img_features: (B, N, T, d_vis)
                - text_features: (B, T, d_text)
                - stage_logits: (B, T, num_classes)
                - progress: (B, T)
        """
        stage_key, prog_key = jr.split(key) if key is not None else (None, None)

        img_features, text_features = self.clip_inference(
            images, text_tokens, img_chunk_size=img_chunk_size
        )

        stage_logits = self.predict_stage(
            img_features, text_features, state, length, dense_schema, stage_key
        )

        # Use predicted stage for progress prediction
        stage_emb = jax.nn.one_hot(
            jnp.argmax(stage_logits, axis=-1),
            num_classes=stage_logits.shape[-1]
        )

        progress = self.predict_progress(
            img_features, text_features, state, stage_emb, length, dense_schema, prog_key
        )

        return {
            "img_features": img_features,
            "text_features": text_features,
            "stage_logits": stage_logits,
            "progress": progress,
        }

    def load_checkpoint(self, progress_checkpoint_path, stage_checkpoint_path):
        assert (progress_checkpoint_path is not None), "Progress checkpoint path is required"
        assert (stage_checkpoint_path is not None), "Stage checkpoint path is required"
        self.progress_transformer = self.progress_transformer.load_checkpoint(progress_checkpoint_path)
        self.stage_transformer = self.stage_transformer.load_checkpoint(stage_checkpoint_path)
        logger.info(f"Loaded checkpoint from {progress_checkpoint_path} and {stage_checkpoint_path}")

    @classmethod
    def init_sarm_from_config(cls, config, key: Array):
        progress_key, stage_key, clip_key = jr.split(key, 3)
        progress_transformer = ProgressTransformer(
            d_model=config.model_config.d_model,
            nheads=config.model_config.n_heads,
            layers=config.model_config.n_layers,
            num_cameras=len(config.general_config.camera_names),
            state_dim=config.model_config.state_dim,
            dropout=config.model_config.dropout,
            key=progress_key,
        )
        logger.info(f"Created SARM progress transformer")
        stage_transformer = StageTransformer(
            d_model=config.model_config.d_model,
            nheads=config.model_config.n_heads,
            layers=config.model_config.n_layers,
            num_cameras=len(config.general_config.camera_names),
            state_dim=config.model_config.state_dim,
            num_classes_sparse=len(config.model_config.sparse_annotation_list),
            dropout=config.model_config.dropout,
            key=stage_key,
        )
        logger.info(f"Created SARM stage transformer")
        clip_model = load_clip_npz(CLIP(key=clip_key), config.model_config.clip_weights_path)
        logger.info(f"Loaded CLIP model from {config.model_config.clip_weights_path}")
        return cls(progress_transformer=progress_transformer,
                   stage_transformer=stage_transformer,
                   clip_model=clip_model)

    def save_model(self, config, step):
        datetime_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
        self.progress_transformer.save_checkpoint(
            f"checkpoints/prg_t-{datetime_str}-s-{step}-b{config.train_loader_config.batch_size}.eqx"
        )
        self.stage_transformer.save_checkpoint(
            f"checkpoints/stg_t-{datetime_str}-s-{step}-b{config.train_loader_config.batch_size}.eqx"
        )