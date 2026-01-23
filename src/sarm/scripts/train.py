import multiprocessing
import os
import logging
from dataclasses import dataclass
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
from jaxtyping import PRNGKeyArray
from tqdm import tqdm

import wandb
from sarm.config.sarm_config import SarmConfig
from sarm.dataset.data_utils import get_valid_episodes, split_train_eval_episodes
from sarm.dataset.dataset import SarmDataset
from sarm.model.clip import preprocess_images_batch
from sarm.model.sarm import ProgressTransformer, StageTransformer, Sarm
from sarm.utils.logger_setup import setup_logger

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class PreparedBatch:
    """Container for shared train/eval batch data."""

    img_features: jax.Array
    text_features: jax.Array
    states: jax.Array
    lengths: jax.Array
    gt_stage: jax.Array
    gt_progress: jax.Array
    dense_schema: bool
    B: int
    T: int


class OptState(eqx.Module):
    """Training state holding optimizer-related data and lr schedule."""

    progress_opt_state: optax.OptState
    stage_opt_state: optax.OptState
    step: int

    # Static fields (not traced)
    optimizer: optax.GradientTransformation = eqx.field(static=True)
    lr_schedule: optax.Schedule = eqx.field(static=True)


@eqx.filter_jit
def step_progress_transformer(
    progress_transformer: ProgressTransformer,
    img_features: jax.Array,
    text_features: jax.Array,
    state: jax.Array,
    subtask: jax.Array,
    length: jax.Array,
    dense_schema: bool,
    progress_targets: jax.Array,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    prog_key: PRNGKeyArray,
):
    """Single training step for ProgressTransformer.

    Args:
        progress_transformer (ProgressTransformer): The ProgressTransformer model
        img_features (jax.Array): Shape (B, N, T, d_vis)
        text_features (jax.Array): Shape (B, T, d_text)
        state_features (jax.Array): Shape (B, T, d_state)
        subtasks (jax.Array): Shape (B, T, C)
        dense_schema (bool): Boolean if the schema is dense
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
        progress_targets,
        prog_key,
    ):
        B = img_features.shape[0]
        prog_keys = jr.split(prog_key, B)
        pred_progress = jax.vmap(
            progress_transformer,
            in_axes=(0, 0, 0, 0, 0, None, 0),
        )(
            img_features, text_features, state, subtask, length, dense_schema, prog_keys
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
        progress_targets,
        prog_key,
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
    dense_schema: bool,
    optimizer: optax.GradientTransformation,
    opt_state: optax.OptState,
    stage_key: PRNGKeyArray,
):
    """Single training step for StageTransformer.

    Args:
        stage_transformer (StageTransformer): The StageTransformer model
        img_features (jax.Array): Shape (B, N, T, d_vis)
        text_features (jax.Array): Shape (B, T, d_text)
        state_features (jax.Array): Shape (B, T, d_state)
        subtasks (jax.Array): Shape (B, T, C)
        dense_schema (bool): Boolean if the schema is dense
    """

    @eqx.filter_value_and_grad(has_aux=True)
    def loss_fn(stage_transformer, img_features, text_features, state_features, length, stage_key):
        B = img_features.shape[0]
        stage_keys = jr.split(stage_key, B)
        logits = jax.vmap(
            stage_transformer,
            in_axes=(0, 0, 0, 0, None, 0),
        )(
            img_features, text_features, state_features, length, dense_schema, stage_keys
        )  # (B, T, C)
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits.reshape(-1, logits.shape[-1]),
            labels=subtasks.reshape(-1),
        )  # (B, )
        return jnp.mean(loss), logits

    (loss, logits), grads = loss_fn(
        stage_transformer,
        img_features,
        text_features,
        state_features,
        length,
        stage_key,
    )
    updates, opt_state = optimizer.update(grads, opt_state, stage_transformer)
    stage_transformer = eqx.apply_updates(stage_transformer, updates)

    return stage_transformer, opt_state, loss, grads, logits


def preprocess_batch(batch, sarm_model: Sarm, config: SarmConfig, dense_schema: bool = False):
    B, T = batch[sarm_model.camera_names[0]].shape[:2]

    # Image Preprocessing
    imgs_list = [batch[cam] for cam in sarm_model.camera_names]
    imgs = np.stack(imgs_list, axis=0)  # N, B, T, C, H, W
    imgs_preprocessed = preprocess_images_batch(imgs, chunk_size=config.model_config.clip_preprocess_chunk_size)
    imgs_preprocessed = einops.rearrange(imgs_preprocessed, "n b t c h w -> b n t c h w")

    # Text Preprocessing
    text_str = batch["task"]  # B
    text_tokens = einops.repeat(jnp.array(sarm_model.tokenizer(text_str)), "b s -> b t s", t=T)

    # State Preprocessing
    states = batch["observation.state"]  # B T D
    states = jnp.array(sarm_model.state_normalizer.normalize(states).detach().numpy())

    lengths = jnp.array(batch["lengths"])
    progress = jnp.array(batch["targets"])

    return {
        "imgs_preprocessed": imgs_preprocessed,
        "text_tokens": text_tokens,
        "states": states,
        "dense_schema": dense_schema,
        "lengths": lengths,
        "progress": progress,
        "B": B,
        "T": T,
    }


def prepare_batch(batch: dict, sarm_model: Sarm, config: SarmConfig) -> PreparedBatch:
    preprocessed = preprocess_batch(
        batch=batch,
        sarm_model=sarm_model,
        config=config,
        dense_schema=config.train_config.dense_shema,
    )

    gt_stage = jnp.floor(preprocessed["progress"]).astype(jnp.int32)
    gt_progress = jnp.remainder(preprocessed["progress"], 1.0)

    img_features, text_features = sarm_model.clip_inference(
        preprocessed["imgs_preprocessed"],
        preprocessed["text_tokens"],
        img_chunk_size=config.model_config.clip_preprocess_chunk_size,
    )

    return PreparedBatch(
        img_features=img_features,
        text_features=text_features,
        states=preprocessed["states"],
        lengths=preprocessed["lengths"],
        gt_stage=gt_stage,
        gt_progress=gt_progress,
        dense_schema=preprocessed["dense_schema"],
        B=preprocessed["B"],
        T=preprocessed["T"],
    )


def log_train_metrics(info: dict, step: int, total_steps: int):
    """Log training metrics to logger and wandb."""
    info_logged = {k: (v.item() if hasattr(v, "item") else v) for k, v in info.items()}
    logger.info(
        f"Step {step}/{total_steps} | "
        f"Total Loss: {info_logged['train/total_loss']:.4f} | "
        f"Stage Loss: {info_logged['train/stage_loss']:.4f} | "
        f"Progress Loss: {info_logged['train/progress_loss']:.4f} | "
        f"LR: {info_logged['train/lr']:.2e}"
    )
    wandb.log(info_logged, step=step)


def log_eval_metrics(eval_metrics_list, step):
    avg_metrics = {f"val/{k}": float(np.mean([m[k] for m in eval_metrics_list])) for k in eval_metrics_list[0].keys()}

    logger.info(
        f"Validation Results - "
        f"Total Loss: {avg_metrics['val/total_loss']:.4f} | "
        f"Stage Loss: {avg_metrics['val/stage_loss']:.4f} | "
        f"Progress Loss: {avg_metrics['val/progress_loss']:.4f} | "
        f"Stage Acc: {avg_metrics['val/stage_acc']:.4f} | "
        f"Total MAE: {avg_metrics['val/total_mae']:.4f}"
    )
    wandb.log(avg_metrics, step=step)


def init_wandb(config):
    logger.info(
        f"Initializing wandb for project: {config.general_config.project_name}-{config.general_config.task_name}"  # noqa: E501
    )
    assert "WANDB_API_KEY" in os.environ
    wandb.init(
        project=f"{config.general_config.project_name}-{config.general_config.task_name}",
        name=f'{datetime.now().strftime("%Y.%m.%d-%H.%M.%S")}',
        config=config,  # noqa: E501
        entity=config.general_config.wandb_entity,
    )


def load_dataset_sparse(config):
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
    return train_episodes_sparse, eval_episodes_sparse


def get_train_and_val_dataset_loader(config, train_episodes_sparse, eval_episodes_sparse):

    def worker_init_fn(worker_id):
        """Prevent DataLoader workers from initializing JAX"""
        import os

        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
        os.environ["JAX_PLATFORMS"] = "cpu"

    mp_context = None
    if config.train_loader_config.num_workers > 0:
        mp_context = multiprocessing.get_context("spawn")

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
        worker_init_fn=worker_init_fn,
        drop_last=True,
        multiprocessing_context=mp_context,
        # pin_memory=config.train_loader_config.pin_memory,
        persistent_workers=config.train_loader_config.persistant_workers > 0,
    )  # type: ignore

    # TODO: Separate config for val loader
    val_loader = torch.utils.data.DataLoader(
        dataset=eval_dataset_sparse,
        batch_size=config.train_loader_config.batch_size,
        shuffle=config.train_loader_config.shuffle,
        num_workers=config.train_loader_config.num_workers,
        worker_init_fn=worker_init_fn,
        multiprocessing_context=mp_context,
        persistent_workers=config.train_loader_config.persistant_workers > 0,
        # pin_memory=config.train_loader_config.pin_memory,
        # persistent_workers=config.train_loader_config.persistant_workers,
        drop_last=True,
    )  # type: ignore
    return train_loader, val_loader


def create_optimizer(config: SarmConfig) -> tuple[optax.GradientTransformation, optax.Schedule]:
    """Create optimizer and learning rate schedule from config."""
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.optimizer_config.lr,
        warmup_steps=config.optimizer_config.warmup_steps,
        decay_steps=config.optimizer_config.total_steps,
        end_value=0.0,
    )

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

    return optimizer, lr_schedule


def init_opt_state(
    optimizer: optax.GradientTransformation,
    lr_schedule: optax.Schedule,
    sarm_model: Sarm,
) -> OptState:
    """Initialize training state from optimizer and model."""
    progress_opt_state = optimizer.init(eqx.filter(sarm_model.progress_transformer, eqx.is_inexact_array))
    stage_opt_state = optimizer.init(eqx.filter(sarm_model.stage_transformer, eqx.is_inexact_array))
    return OptState(
        progress_opt_state=progress_opt_state,
        stage_opt_state=stage_opt_state,
        step=0,
        optimizer=optimizer,
        lr_schedule=lr_schedule,
    )


def eval_step(batch: dict, sarm_model: Sarm, config: SarmConfig):
    """Evaluation step - no gradients, inference mode."""
    # Put entire model in inference mode (handles dropout, etc.)
    sarm_model_eval = eqx.nn.inference_mode(sarm_model)

    prepared = prepare_batch(batch=batch, sarm_model=sarm_model_eval, config=config)

    # Stage prediction (inference mode)
    logits = sarm_model_eval.predict_stage(
        img_features=prepared.img_features,
        text_features=prepared.text_features,
        state=prepared.states,
        length=prepared.lengths,
        dense_schema=prepared.dense_schema,
        key=None,
    )

    # Use predicted stage for progress
    stage_emb = jax.nn.one_hot(jnp.argmax(logits, axis=-1), num_classes=logits.shape[-1])

    # Progress prediction (inference mode)
    pred_progress = sarm_model_eval.predict_progress(
        img_features=prepared.img_features,
        text_features=prepared.text_features,
        state=prepared.states,
        stage_emb=stage_emb,
        length=prepared.lengths,
        dense_schema=prepared.dense_schema,
        key=None,
    )

    # Compute losses
    stage_loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits.reshape(-1, logits.shape[-1]),
        labels=prepared.gt_stage.reshape(-1),
    )
    stage_loss = jnp.mean(stage_loss)

    progress_loss = jnp.mean(jnp.square(pred_progress - prepared.gt_progress))
    total_loss = stage_loss + progress_loss

    # Additional metrics
    stage_acc = jnp.mean((jnp.argmax(logits, axis=-1) == prepared.gt_stage).astype(jnp.float32))
    progress_mae = jnp.mean(jnp.abs(pred_progress - prepared.gt_progress))

    # Combined prediction MAE
    pred_combined = jnp.argmax(logits, axis=-1).astype(jnp.float32) + pred_progress
    gt_combined = prepared.gt_stage.astype(jnp.float32) + prepared.gt_progress
    total_mae = jnp.mean(jnp.abs(pred_combined - gt_combined))

    return {
        "stage_loss": float(stage_loss),
        "progress_loss": float(progress_loss),
        "total_loss": float(total_loss),
        "stage_acc": float(stage_acc),
        "progress_mae": float(progress_mae),
        "total_mae": float(total_mae),
    }


def train_step(
    batch: dict,
    sarm_model: Sarm,
    opt_state: OptState,
    config: SarmConfig,
    train_key: PRNGKeyArray,
):
    train_key, stage_key, prog_key = jr.split(train_key, 3)

    batch_pre = prepare_batch(batch=batch, sarm_model=sarm_model, config=config)

    # Stage transformer training step
    stage_transformer, stage_opt_state, stage_loss, stage_grads, logits = step_stage_transformer(
        sarm_model.stage_transformer,
        batch_pre.img_features,
        batch_pre.text_features,
        batch_pre.states,
        batch_pre.gt_stage,
        batch_pre.lengths,
        batch_pre.dense_schema,
        opt_state.optimizer,
        opt_state.stage_opt_state,
        stage_key,
    )

    # Stage embedding with teacher forcing (75% GT, 25% predicted)
    if torch.rand(1).item() < 0.75:
        stage_emb = jax.nn.one_hot(
            batch_pre.gt_stage,
            num_classes=len(config.model_config.sparse_annotation_list),
        )
    else:
        stage_emb = jax.nn.one_hot(jnp.argmax(logits, axis=-1), num_classes=logits.shape[-1], axis=-1)

    # Progress transformer training step
    progress_transformer, progress_opt_state, progress_loss, progress_grads = step_progress_transformer(
        sarm_model.progress_transformer,
        batch_pre.img_features,
        batch_pre.text_features,
        batch_pre.states,
        stage_emb,
        batch_pre.lengths,
        batch_pre.dense_schema,
        batch_pre.gt_progress,
        opt_state.optimizer,
        opt_state.progress_opt_state,
        prog_key,
    )

    # Stage prediction quality
    stage_preds = jnp.argmax(logits, axis=-1)
    stage_accuracy = float(jnp.mean((stage_preds == batch_pre.gt_stage).astype(jnp.float32)))

    info = {
        "train/stage_loss": stage_loss,
        "train/progress_loss": progress_loss,
        "train/total_loss": (stage_loss + progress_loss),
        "train/stage_grad_norm": optax.global_norm(stage_grads),
        "train/progress_grad_norm": optax.global_norm(progress_grads),
        "train/lr": opt_state.lr_schedule(opt_state.step),
        "train/stage_accuracy": stage_accuracy,
    }

    # Update model with trained transformers
    new_model = eqx.tree_at(
        lambda m: (m.stage_transformer, m.progress_transformer),
        sarm_model,
        (stage_transformer, progress_transformer),
    )

    # Update optimizer state
    new_opt_state = OptState(
        progress_opt_state=progress_opt_state,
        stage_opt_state=stage_opt_state,
        step=opt_state.step + 1,
        optimizer=opt_state.optimizer,
        lr_schedule=opt_state.lr_schedule,
    )

    return new_model, new_opt_state, train_key, info


def get_next_batch(data_iter, data_loader):
    try:
        batch = next(data_iter)
    except StopIteration:
        train_iter = iter(data_loader)
        batch = next(train_iter)
    return batch


def train(config: SarmConfig):
    if config.train_config.dense_shema:
        raise NotImplementedError("Dense schema training is not implemented yet")

    setup_logger(config, logger)
    init_wandb(config)
    train_episodes_sparse, eval_episodes_sparse = load_dataset_sparse(config)
    train_loader, val_loader = get_train_and_val_dataset_loader(
        config,
        train_episodes_sparse=train_episodes_sparse,
        eval_episodes_sparse=eval_episodes_sparse,
    )
    train_iter = iter(train_loader)
    eval_iter = iter(val_loader)
    logger.info("Initialized data loader")

    model_key, train_key = jr.split(jr.PRNGKey(config.general_config.seed), 2)

    sarm_model = Sarm.init_sarm_from_config(config=config, key=model_key)
    logger.info("Initialized models")

    if config.model_config.resume_from_checkpoint:
        sarm_model.load_checkpoint(
            stage_checkpoint_path=config.model_config.stage_checkpoint_path,
            progress_checkpoint_path=config.model_config.progress_checkpoint_path,
        )

    logger.info(
        f"Setting up optimizer with lr={config.optimizer_config.lr}, "
        f"warmup_steps={config.optimizer_config.warmup_steps}"
    )

    optimizer, lr_schedule = create_optimizer(config)
    opt_state = init_opt_state(optimizer, lr_schedule, sarm_model)
    logger.info("Optimizer states initialized")

    # TRAIN LOOP
    for _ in (pbar := tqdm(range(config.optimizer_config.total_steps), desc="Training", unit="step")):
        batch = get_next_batch(train_iter, train_loader)

        sarm_model, opt_state, train_key, info = train_step(
            batch=batch,
            sarm_model=sarm_model,
            opt_state=opt_state,
            config=config,
            train_key=train_key,
        )
        pbar.set_postfix({"total_loss": f"{info['train/total_loss']:.4f}", "lr": f"{info['train/lr']:.2e}"})

        if opt_state.step % config.train_config.log_every == 0:
            log_train_metrics(info, opt_state.step, config.optimizer_config.total_steps)

        if opt_state.step % config.train_config.eval_every == 0 and opt_state.step > 0:
            logger.info("Evaluating on validation set...")
            eval_metrics_list = []
            num_eval_batches = min(10, len(val_loader))

            for _ in range(num_eval_batches):
                eval_batch = get_next_batch(eval_iter, val_loader)
                metrics = eval_step(batch=eval_batch, sarm_model=sarm_model, config=config)
                eval_metrics_list.append(metrics)
            log_eval_metrics(eval_metrics_list=eval_metrics_list, step=opt_state.step)

        if (
            opt_state.step % config.train_config.save_every == 0
            or opt_state.step == config.optimizer_config.total_steps
        ):
            sarm_model.save_model(step=opt_state.step, config=config)

    logger.info("Training completed!")


if __name__ == "__main__":
    config = SarmConfig()
    train(config)
