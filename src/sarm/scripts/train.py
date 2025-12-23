import logging
from datetime import datetime

import einops
import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import torch
from dotenv import load_dotenv
from tqdm import tqdm

import wandb
from sarm.config.sarm_config import SarmConfig
from sarm.dataset.data_utils import get_valid_episodes, split_train_eval_episodes
from sarm.dataset.dataset import SarmDataset
from sarm.dataset.normalizer import get_normalizer_from_calculated
from sarm.model.clip import CLIP, load_clip_npz, preprocess_images_batch
from sarm.model.sarm import ProgressTransformer, StageTransformer
from sarm.utils.logging import setup_logger
from sarm.utils.tokenizer import load_tokenizer

load_dotenv()

logger = logging.getLogger(__name__)


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
def step_progress_transformer(
    progress_transformer: ProgressTransformer,
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
        progress_transformer (ProgressTransformer): The ProgressTransformer model
        img_features (jax.Array): Shape (B, N, T, d_vis)
        text_features (jax.Array): Shape (B, T, d_text)
        state_features (jax.Array): Shape (B, T, d_state)
        subtasks (jax.Array): Shape (B, T, C)
        dense_schema (jax.Array): Shape (B,)
        progress_targets (jax.Array): Shape (B, T)
    """

    @eqx.filter_value_and_grad
    def loss_fn(
        progress_transformer,
        img_features,
        text_features,
        state,
        subtask,
        length,
        dense_schema,
        progress_targets,
    ):
        pred_progress = jax.vmap(
            progress_transformer,
            in_axes=(0, 0, 0, 0, 0, 0),
        )(
            img_features, text_features, state, subtask, length, dense_schema
        )  # (B, T)

        loss = jnp.mean(jnp.square(pred_progress - progress_targets))
        return loss

    loss, grads = loss_fn(
        progress_transformer,
        img_features,
        text_features,
        state,
        subtask,
        length,
        dense_schema,
        progress_targets,
    )
    updates, opt_state = optimizer.update(grads, opt_state, progress_transformer)
    progress_transformer = eqx.apply_updates(progress_transformer, updates)

    return progress_transformer, opt_state, loss, grads


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
            labels=subtasks.reshape(-1),
        )  # (B, )
        return jnp.mean(loss), logits

    (loss, logits), grads = loss_fn(
        stage_transformer, img_features, text_features, state_features, length, dense_schema
    )
    updates, opt_state = optimizer.update(grads, opt_state, stage_transformer)
    stage_transformer = eqx.apply_updates(stage_transformer, updates)

    return stage_transformer, opt_state, loss, grads, logits


def gen_stage_emb(self, num_classes, trg):
    """
    Returns stage_onehot with a modality dim (B, 1, T, C).
    """
    # integer part of float targets -> [0, C-1]
    idx = trg.long().clamp(min=0, max=num_classes - 1)  # (B, T)
    C = num_classes
    # identity-lookup one-hot
    stage_onehot = torch.eye(C, device=trg.device)[idx]  # (B, T, C)
    stage_onehot = stage_onehot.unsqueeze(1)  # (B, 1, T, C)
    return stage_onehot


def train(config: SarmConfig):

    # Setup logger first
    setup_logger(config, logger)

    # Set multiprocessing start method to avoid JAX fork() issues
    # Must be called before creating DataLoaders with num_workers > 0
    torch.multiprocessing.set_start_method("spawn", force=True)

    logger.info(
        f"Initializing wandb for project: {config.general_config.project_name}-{config.general_config.task_name}"  # noqa: E501
    )
    wandb.init(
        project=f"{config.general_config.project_name}-{config.general_config.task_name}",
        name=f'{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}',
        config=config,  # noqa: E501
        entity=config.general_config.wandb_entity,
    )

    ###############################################################################################
    #                                       Load Datasets                                         #
    ###############################################################################################

    logger.info("Loading datasets...")
    valid_episodes_sparse = get_valid_episodes(config.general_config.repo_id_sparse)
    logger.info(f"Found {len(valid_episodes_sparse)} valid sparse episodes")
    train_episodes_sparse, eval_episodes_sparse = split_train_eval_episodes(
        valid_episodes_sparse,
        1 - config.train_config.val_portion,
        seed=config.general_config.seed,
    )
    logger.info(
        f"Split into {len(train_episodes_sparse)} train and {len(eval_episodes_sparse)} eval episodes"  # noqa: E501
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
        # pin_memory=config.train_loader_config.pin_memory,
        # persistent_workers=config.train_loader_config.persistant_workers,
    )  # type: ignore

    # TODO: Separate config for val loader
    val_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset_sparse,
        batch_size=config.train_loader_config.batch_size,
        shuffle=config.train_loader_config.shuffle,
        num_workers=config.train_loader_config.num_workers,
        # pin_memory=config.train_loader_config.pin_memory,
        # persistent_workers=config.train_loader_config.persistant_workers,
    )  # type: ignore

    state_normalizer = get_normalizer_from_calculated(config.general_config.state_norm_path, "cpu")
    logger.info(f"Loaded state normalizer from {config.general_config.state_norm_path}")

    tokenizer = load_tokenizer()
    logger.info("Loaded tokenizer")

    ###############################################################################################
    #                                    Initialize Modules                                       #
    ###############################################################################################

    logger.info("Initializing models...")
    progress_key, stage_key, clip_key = jr.split(jr.PRNGKey(config.general_config.seed), 3)

    progress_transformer = ProgressTransformer(
        d_model=config.model_config.d_model,
        nheads=config.model_config.n_heads,
        layers=config.model_config.n_layers,
        num_cameras=len(config.general_config.camera_names),
        state_dim=config.model_config.state_dim,
        key=progress_key,
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
    logger.info(f"Loaded CLIP model from {config.model_config.clip_weights_path}")

    if config.model_config.resume_from_checkpoint:
        assert (
            config.model_config.progress_checkpoint_path is not None
        ), "Progress checkpoint path is required"
        assert (
            config.model_config.stage_checkpoint_path is not None
        ), "Stage checkpoint path is required"
        progress_transformer.load_checkpoint(config.model_config.progress_checkpoint_path)
        stage_transformer.load_checkpoint(config.model_config.stage_checkpoint_path)
        logger.info(
            f"Loaded checkpoint from {config.model_config.progress_checkpoint_path} and {config.model_config.stage_checkpoint_path}"  # noqa: E501
        )

    # Learning rate schedule: linear warmup + cosine annealing
    logger.info(
        f"Setting up optimizer with lr={config.optimizer_config.lr}, warmup_steps={config.optimizer_config.warmup_steps}"  # noqa: E501
    )
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optimizer_config.lr,
        warmup_steps=config.optimizer_config.warmup_steps,
        decay_steps=config.optimizer_config.total_steps,
        end_value=0.0,
    )

    # Optimizer with gradient clipping
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.train_config.grad_clip),
        optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=config.optimizer_config.weight_decay,
            b1=config.optimizer_config.betas[0],
            b2=config.optimizer_config.betas[1],
            eps=config.optimizer_config.eps,
        ),
    )

    progress_opt_state = optimizer.init(eqx.filter(progress_transformer, eqx.is_inexact_array))
    stage_opt_state = optimizer.init(eqx.filter(stage_transformer, eqx.is_inexact_array))
    logger.info("Optimizer states initialized")

    def train_step(
        batch: dict,
        progress_transformer: ProgressTransformer,
        stage_transformer: StageTransformer,
        progress_opt_state: optax.OptState,
        stage_opt_state: optax.OptState,
        step: int,
    ):
        B, T = batch[config.general_config.camera_names[0]].shape[:2]

        # Image Preprocessing
        imgs_list = []
        for key in config.general_config.camera_names:
            imgs_list.append(batch[key])
        imgs = np.stack(imgs_list, axis=0)  # N, B, T, C, H, W
        imgs_preprocessed = preprocess_images_batch(imgs)
        imgs_preprocessed = einops.rearrange(imgs_preprocessed, "n b t c h w -> b n t c h w")

        # Text Preprocessing
        text_str = batch["task"]  # B
        text_tokens = einops.repeat(jnp.array(tokenizer(text_str)), "b s -> b t s", t=T)

        # State Preprocessing
        states = batch["observation.state"]  # B T D
        states = jnp.array(state_normalizer.normalize(states).detach().numpy())

        dense_schemas = jnp.array([False for _ in range(B)])
        lengths = jnp.array(batch["lengths"])
        progress = jnp.array(batch["targets"])
        gt_stage = jnp.floor(progress).astype(jnp.int32)
        gt_progress = jnp.remainder(progress, 1.0)

        img_features, text_features = clip_inference(clip_model, imgs_preprocessed, text_tokens)

        stage_transformer, stage_opt_state, stage_loss, stage_grads, logits = (
            step_stage_transformer(
                stage_transformer,
                img_features,
                text_features,
                states,
                gt_stage,
                lengths,
                dense_schemas,
                optimizer,
                stage_opt_state,
            )
        )

        if torch.rand(1).item() < 0.75:
            # Mode 1: ground truth trg → one-hot
            stage_emb = jax.nn.one_hot(
                gt_stage, num_classes=len(config.model_config.sparse_annotation_list)
            )
        else:
            # Mode 2: predicted argmax → one-hot
            stage_emb = jax.nn.one_hot(
                jnp.argmax(logits, axis=-1), num_classes=logits.shape[-1], axis=-1
            )

        progress_transformer, progress_opt_state, progress_loss, progress_grads = (
            step_progress_transformer(
                progress_transformer,
                img_features,
                text_features,
                states,
                stage_emb,
                lengths,
                dense_schemas,
                gt_progress,
                optimizer,
                progress_opt_state,
            )
        )

        info = {
            "train/stage_loss": stage_loss.item(),
            "train/progress_loss": progress_loss.item(),
            "train/total_loss": (stage_loss + progress_loss).item(),
            "train/stage_grad_norm": optax.global_norm(stage_grads).item(),
            "train/progress_grad_norm": optax.global_norm(progress_grads).item(),
            "train/lr": lr_schedule(step).item(),
        }

        return stage_transformer, progress_transformer, progress_opt_state, stage_opt_state, info

    step = 0
    train_iter = iter(train_loader)

    logger.info(f"Starting training for {config.optimizer_config.total_steps} steps...")

    # Create progress bar
    pbar = tqdm(total=config.optimizer_config.total_steps, desc="Training", unit="step")

    while True:
        batch = next(train_iter)
        stage_transformer, progress_transformer, progress_opt_state, stage_opt_state, info = (
            train_step(
                batch,
                progress_transformer,
                stage_transformer,
                progress_opt_state,
                stage_opt_state,
                step=step,
            )
        )

        # Update progress bar with metrics
        pbar.set_postfix(
            {"total_loss": f"{info['train/total_loss']:.4f}", "lr": f"{info['train/lr']:.2e}"}
        )
        pbar.update(1)

        # Log training metrics (less frequently to avoid clutter)
        if step % config.train_config.log_every == 0:
            logger.info(
                f"Step {step}/{config.optimizer_config.total_steps} | "
                f"Total Loss: {info['train/total_loss']:.4f} | "
                f"Stage Loss: {info['train/stage_loss']:.4f} | "
                f"Progress Loss: {info['train/progress_loss']:.4f} | "
                f"LR: {info['train/lr']:.2e}"
            )
            wandb.log(info)

        step += 1

        if (
            step % config.train_config.save_every == 0
            or step == config.optimizer_config.total_steps
        ):
            datetime_str = datetime.now().strftime("%Y.%m.%d-%H.%M.%S")
            progress_transformer.save_checkpoint(
                f"checkpoints/prg_t-{datetime_str}-s-{step}-b{config.train_loader_config.batch_size}.eqx"
            )
            stage_transformer.save_checkpoint(
                f"checkpoints/stg_t-{datetime_str}-s-{step}-b{config.train_loader_config.batch_size}.eqx"
            )

        if step == config.optimizer_config.total_steps:
            pbar.close()
            logger.info("Training completed!")
            break


if __name__ == "__main__":
    config = SarmConfig()
    train(config)
