import logging

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr

from sarm.model.clip import CLIP, Block

logger = logging.getLogger(__name__)


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
        key=jr.PRNGKey(0),
    ):
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

        self.blocks = [Block(d_model, nheads, key=jr.fold_in(k_blocks, i)) for i in range(layers)]
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
        for block in self.blocks:
            features = block(features, mask)

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
        self = eqx.tree_deserialise_leaves(path, self)
        logger.info(f"Loaded checkpoint from {path}")


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
        key=jr.PRNGKey(0),
    ):
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

        self.blocks = [Block(d_model, nheads, key=jr.fold_in(k_blocks, i)) for i in range(layers)]
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
        for block in self.blocks:
            features = block(features, mask)

        features = einops.rearrange(features, "(n t) d -> t (n d)", n=N + 2, t=T)
        features = jax.vmap(self.fusion_mlp)(features)  # (T, d_model)

        logits = jax.vmap(self.final_proj["sparse"])(
            features
        )  # (T, C) TODO: add conditional sparse projection

        return logits  # (T, C)

    def save_checkpoint(self, path: str):
        eqx.tree_serialise_leaves(path, self)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        self = eqx.tree_deserialise_leaves(path, self)
        logger.info(f"Loaded checkpoint from {path}")


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
