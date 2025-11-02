import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import torch

from sarm.config.sarm_config import SarmConfig
from sarm.dataset.data_utils import get_valid_episodes, split_train_eval_episodes
from sarm.dataset.dataset import SarmDataset
from sarm.model.clip import CLIP, load_clip_npz
from sarm.model.sarm import ProgressTransformer, StageTransformer


@eqx.filter_jit
def clip_inference(
    clip_model: CLIP,
    images: jax.Array,
    text_tokens: jax.Array,
):
    """Extract features using CLIP model.

    Args:
        clip_model (CLIP): The CLIP model
        images (jax.Array): Shape (B, N, T, C, H, W)
        text_tokens (jax.Array): Shape (B, T, max_len)

    Returns:
        img_features (jax.Array): Shape (B, N, T, d_vis)
        text_features (jax.Array): Shape (B, T, d_text)
    """
    B, N, T, C, H, W = images.shape
    images_reshaped = images.reshape((B * N * T, C, H, W))
    img_features = jax.vmap(clip_model.encode_image)(images_reshaped)
    img_features = img_features.reshape((B, N, T, -1))  # (B, N, T, d_vis)

    text_features = jax.vmap(jax.vmap(clip_model.encode_text))(text_tokens)  # (B, T, d_text)
    return img_features, text_features


@eqx.filter_jit
def step_process_transformer(
    process_transformer: ProgressTransformer,
    img_features: jax.Array,
    text_features: jax.Array,
    state: jax.Array,
    subtask: jax.Array,
    length: jax.Array,
    dense_schema: jax.Array,
    progress_targets: jax.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    """Single training step for ProgressTransformer.

    Args:
        process_transformer (ProgressTransformer): The ProgressTransformer model
        img_features (jax.Array): Shape (B, N, T, d_vis)
        text_features (jax.Array): Shape (B, T, d_text)
        state_features (jax.Array): Shape (B, T, d_state)
        subtasks (jax.Array): Shape (B, T, C)
        dense_schema (jax.Array): Shape (B,)
        progress_targets (jax.Array): Shape (B, T)
    """

    @eqx.filter_value_and_grad
    def loss_fn(
        process_transformer,
        img_features,
        text_features,
        state,
        subtask,
        length,
        dense_schema,
        progress_targets,
    ):
        pred_progress = jax.vmap(
            process_transformer,
            in_axes=(0, 0, 0, 0, 0, 0),
        )(
            img_features, text_features, state, subtask, length, dense_schema
        )  # (B, T)

        loss = jnp.mean(jnp.square(pred_progress - progress_targets))
        return loss

    loss, grads = loss_fn(
        process_transformer,
        img_features,
        text_features,
        state,
        subtask,
        length,
        dense_schema,
        progress_targets,
    )
    updates, opt_state = optimizer.update(grads, opt_state, process_transformer)
    process_transformer = eqx.apply_updates(process_transformer, updates)

    return process_transformer, opt_state, loss, grads


@eqx.filter_jit
def step_stage_transformer(
    stage_transformer: StageTransformer,
    img_features: jax.Array,
    text_features: jax.Array,
    state_features: jax.Array,
    subtasks: jax.Array,
    length: jax.Array,
    dense_schema: jax.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
):
    """Single training step for StageTransformer.

    Args:
        stage_transformer (StageTransformer): The StageTransformer model
        img_features (jax.Array): Shape (B, N, T, d_vis)
        text_features (jax.Array): Shape (B, T, d_text)
        state_features (jax.Array): Shape (B, T, d_state)
        subtasks (jax.Array): Shape (B, T, C)
        dense_schema (jax.Array): Shape (B,)
    """

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(
        stage_transformer, img_features, text_features, state_features, length, dense_schema
    ):
        logits = jax.vmap(
            stage_transformer,
            in_axes=(0, 0, 0, 0, 0),
        )(
            img_features, text_features, state_features, length, dense_schema
        )  # (B, T, C)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits.reshape(-1, logits.shape[-1]),
            labels=subtasks,
        )  # (B, )
        return jnp.mean(loss), logits

    (loss, logits), grads = loss_fn(
        stage_transformer, img_features, text_features, state_features, length, dense_schema
    )
    updates, opt_state = optimizer.update(grads, opt_state, stage_transformer)
    stage_transformer = eqx.apply_updates(stage_transformer, updates)

    return stage_transformer, opt_state, loss, grads, logits


def train(config: SarmConfig):

    ###############################################################################################
    #                                       Load Datasets                                         #
    ###############################################################################################

    valid_episodes_sparse = get_valid_episodes(config.general_config.repo_id_sparse)
    train_episodes_sparse, eval_episodes_sparse = split_train_eval_episodes(
        valid_episodes_sparse,
        1 - config.train_config.val_portion,
        seed=config.general_config.seed,
    )
    train_dataset_sparse = SarmDataset(
        repo_id=config.general_config.repo_id_sparse,
        horizon=config.model_config.horizon,
        episodes=train_episodes_sparse,
        n_obs_steps=config.model_config.n_obs_steps,
        frame_gap=config.model_config.frame_gap,
        max_rewind_steps=config.model_config.max_rewind_steps,
        image_names=config.general_config.camera_names,
        annotation_list=config.model_config.sparse_annotation_list,
        task=config.general_config.task_name,
    )
    eval_dataset_sparse = SarmDataset(
        repo_id=config.general_config.repo_id_sparse,
        horizon=config.model_config.horizon,
        episodes=eval_episodes_sparse,
        n_obs_steps=config.model_config.n_obs_steps,
        frame_gap=config.model_config.frame_gap,
        max_rewind_steps=config.model_config.max_rewind_steps,
        image_names=config.general_config.camera_names,
        annotation_list=config.model_config.sparse_annotation_list,
        task=config.general_config.task_name,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_sparse,
        batch_size=config.train_loader_config.batch_size,
        shuffle=config.train_loader_config.shuffle,
        num_workers=config.train_loader_config.num_workers,
        pin_memory=config.train_loader_config.pin_memory,
        persistent_workers=config.train_loader_config.persistant_workers,
    )  # type: ignore

    # TODO: Separate config for val loader
    val_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset_sparse,
        batch_size=config.train_loader_config.batch_size,
        shuffle=config.train_loader_config.shuffle,
        num_workers=config.train_loader_config.num_workers,
        pin_memory=config.train_loader_config.pin_memory,
        persistent_workers=config.train_loader_config.persistant_workers,
    )  # type: ignore

    ###############################################################################################
    #                                    Initialize Modules                                       #
    ###############################################################################################

    process_key, stage_key, clip_key = jr.split(jr.PRNGKey(config.general_config.seed), 3)

    process_transformer = ProgressTransformer(
        d_model=config.model_config.d_model,
        nheads=config.model_config.n_heads,
        layers=config.model_config.n_layers,
        num_cameras=len(config.general_config.camera_names),
        state_dim=config.model_config.state_dim,
        key=process_key,
    )
    stage_transformer = StageTransformer(
        d_model=config.model_config.d_model,
        nheads=config.model_config.n_heads,
        layers=config.model_config.n_layers,
        num_cameras=len(config.general_config.camera_names),
        state_dim=config.model_config.state_dim,
        num_classes_sparse=len(config.model_config.sparse_annotation_list),
        key=stage_key,
    )
    clip_model = load_clip_npz(CLIP(key=clip_key), config.model_config.clip_weights_path)

    # TODO: Add option to resume training
    



if __name__ == "__main__":
    config = SarmConfig()
    train(config)
