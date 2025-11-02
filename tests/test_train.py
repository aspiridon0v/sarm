import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
import pytest
from torch.utils.data import DataLoader, Dataset

from sarm.model.clip import CLIP, load_tokenizer
from sarm.model.sarm import ProgressTransformer, StageTransformer
from sarm.scripts.train import (
    clip_inference,
    step_process_transformer,
    step_stage_transformer,
)


class DummyDataset(Dataset):
    """Dataset that generates synthetic data for testing."""

    def __init__(self, timesteps=8, num_cameras=1, size=100):
        super().__init__()
        self.tokenizer = load_tokenizer()
        self.timesteps = timesteps
        self.num_cameras = num_cameras
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # NOTE: Currently hardcoded for dense schema
        dense_schema = np.array(True)
        subtask_idx = 2
        subtask = np.zeros((self.timesteps, 8))
        subtask[:, subtask_idx] = 1.0

        # Generate non-zero data for more realistic gradients
        images = np.random.randn(self.num_cameras, self.timesteps, 3, 224, 224) * 0.1
        state = np.random.randn(self.timesteps, 14) * 0.1
        progress = np.linspace(0, 1, self.timesteps)
        text_tokens = np.stack(
            [self.tokenizer("dummy text for testing").squeeze(0) for t in range(self.timesteps)]
        )
        return {
            "img": images.astype(np.float32),
            "text": text_tokens,
            "state": state.astype(np.float32),
            "subtask": subtask.astype(np.float32),
            "dense_schema": dense_schema.astype(np.bool_),
            "length": self.timesteps,
            "progress_target": progress.astype(np.float32),
        }


@pytest.fixture
def dummy_batch():
    """Create a single batch for testing."""
    dataset = DummyDataset(timesteps=8, num_cameras=1, size=4)
    batch_data = [dataset[i] for i in range(4)]

    # Collate batch
    batch = {
        "img": np.stack([d["img"] for d in batch_data]),
        "text": np.stack([d["text"] for d in batch_data]),
        "state": np.stack([d["state"] for d in batch_data]),
        "subtask": np.stack([d["subtask"] for d in batch_data]),
        "dense_schema": np.stack([d["dense_schema"] for d in batch_data]),
        "length": np.array([d["length"] for d in batch_data]),
        "progress_target": np.stack([d["progress_target"] for d in batch_data]),
    }
    return batch


@pytest.fixture
def sarm_modules():
    """Initialize SARM modules."""
    process_key, stage_key, clip_key = jr.split(jr.PRNGKey(42), 3)

    process_transformer = ProgressTransformer(key=process_key)
    stage_transformer = StageTransformer(key=stage_key)
    clip_model = CLIP(key=clip_key)
    return process_transformer, stage_transformer, clip_model


def test_clip_inference_shapes(sarm_modules, dummy_batch):
    """Test that CLIP inference produces correct output shapes."""
    _, _, clip_model = sarm_modules

    images = jnp.array(dummy_batch["img"])  # (B, N, T, C, H, W)
    texts = jnp.array(dummy_batch["text"])  # (B, T, max_len)

    img_features, text_features = clip_inference(clip_model, images, texts)

    B, N, T = images.shape[:3]
    assert img_features.shape[:3] == (
        B,
        N,
        T,
    ), f"Expected shape (B={B}, N={N}, T={T}, d_vis), got {img_features.shape}"
    assert text_features.shape[:2] == (
        B,
        T,
    ), f"Expected shape (B={B}, T={T}, d_text), got {text_features.shape}"
    assert img_features.dtype in [jnp.float32, jnp.float16]
    assert text_features.dtype in [jnp.float32, jnp.float16]


def test_process_transformer_step(sarm_modules, dummy_batch):
    """Test a single training step for ProgressTransformer."""
    process_transformer, _, clip_model = sarm_modules

    # Prepare batch
    images = jnp.array(dummy_batch["img"])
    texts = jnp.array(dummy_batch["text"])
    states = jnp.array(dummy_batch["state"])
    subtasks = jnp.array(dummy_batch["subtask"])
    dense_schemas = jnp.array(dummy_batch["dense_schema"])
    lengths = jnp.array(dummy_batch["length"])
    progress_targets = jnp.array(dummy_batch["progress_target"])

    # Extract features
    img_features, text_features = clip_inference(clip_model, images, texts)

    # Setup optimizer
    optimizer = optax.adamw(learning_rate=1e-4)
    opt_state = optimizer.init(eqx.filter(process_transformer, eqx.is_inexact_array))

    # Training step
    new_model, new_opt_state, loss, grads = step_process_transformer(
        process_transformer,
        img_features,
        text_features,
        states,
        subtasks,
        lengths,
        dense_schemas,
        progress_targets,
        optimizer,
        opt_state,
    )

    # Assertions
    assert isinstance(loss, jax.Array), "Loss should be a JAX array"
    assert loss.shape == (), "Loss should be a scalar"
    assert jnp.isfinite(loss), "Loss should be finite"
    assert loss >= 0, "Loss should be non-negative"

    # Check gradients exist and are non-zero
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(grad_leaves) > 0, "Gradients should be computed"
    grad_norms = [jnp.linalg.norm(g.flatten()) for g in grad_leaves if isinstance(g, jax.Array)]
    assert any(norm > 0 for norm in grad_norms), "At least some gradients should be non-zero"

    # Check model was updated
    old_params = eqx.filter(process_transformer, eqx.is_inexact_array)
    new_params = eqx.filter(new_model, eqx.is_inexact_array)
    param_diff = jax.tree_util.tree_map(lambda x, y: jnp.abs(x - y).sum(), old_params, new_params)
    total_diff = sum(jax.tree_util.tree_leaves(param_diff))
    assert total_diff > 0, "Model parameters should be updated"


def test_stage_transformer_step(sarm_modules, dummy_batch):
    """Test a single training step for StageTransformer."""
    _, stage_transformer, clip_model = sarm_modules

    # Prepare batch
    images = jnp.array(dummy_batch["img"])
    texts = jnp.array(dummy_batch["text"])
    states = jnp.array(dummy_batch["state"])
    subtasks = jnp.array(dummy_batch["subtask"])
    dense_schemas = jnp.array(dummy_batch["dense_schema"])
    lengths = jnp.array(dummy_batch["length"])

    subtask_labels = jnp.argmax(subtasks, axis=-1).reshape(-1).astype(jnp.int32)

    # Extract features
    img_features, text_features = clip_inference(clip_model, images, texts)

    # Setup optimizer
    optimizer = optax.adamw(learning_rate=1e-4)
    opt_state = optimizer.init(eqx.filter(stage_transformer, eqx.is_inexact_array))

    # Training step
    new_model, new_opt_state, loss, grads, _ = step_stage_transformer(
        stage_transformer,
        img_features,
        text_features,
        states,
        subtask_labels,
        lengths,
        dense_schemas,
        optimizer,
        opt_state,
    )

    # Assertions
    assert isinstance(loss, jax.Array), "Loss should be a JAX array"
    assert loss.shape == (), "Loss should be a scalar"
    assert jnp.isfinite(loss), "Loss should be finite"
    assert loss >= 0, "Loss should be non-negative"

    # Check gradients
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(grad_leaves) > 0, "Gradients should be computed"
    grad_norms = [jnp.linalg.norm(g.flatten()) for g in grad_leaves if isinstance(g, jax.Array)]
    assert any(norm > 0 for norm in grad_norms), "At least some gradients should be non-zero"

    # Check model was updated
    old_params = eqx.filter(stage_transformer, eqx.is_inexact_array)
    new_params = eqx.filter(new_model, eqx.is_inexact_array)
    param_diff = jax.tree_util.tree_map(lambda x, y: jnp.abs(x - y).sum(), old_params, new_params)
    total_diff = sum(jax.tree_util.tree_leaves(param_diff))
    assert total_diff > 0, "Model parameters should be updated"


def test_process_transformer_overfitting(sarm_modules):
    """Test that ProgressTransformer can overfit to a small batch (sanity check)."""
    process_transformer, _, clip_model = sarm_modules

    # Create a small fixed batch
    dataset = DummyDataset(timesteps=8, num_cameras=1, size=2)
    batch_data = [dataset[0], dataset[0]]  # Same sample twice for easier overfitting

    batch = {
        "img": np.stack([d["img"] for d in batch_data]),
        "text": np.stack([d["text"] for d in batch_data]),
        "state": np.stack([d["state"] for d in batch_data]),
        "subtask": np.stack([d["subtask"] for d in batch_data]),
        "dense_schema": np.stack([d["dense_schema"] for d in batch_data]),
        "length": np.array([d["length"] for d in batch_data]),
        "progress_target": np.stack([d["progress_target"] for d in batch_data]),
    }

    # Prepare batch
    images = jnp.array(batch["img"])
    texts = jnp.array(batch["text"])
    states = jnp.array(batch["state"])
    subtasks = jnp.array(batch["subtask"])
    dense_schemas = jnp.array(batch["dense_schema"])
    lengths = jnp.array(batch["length"])
    progress_targets = jnp.array(batch["progress_target"])

    # Extract features once (fixed)
    img_features, text_features = clip_inference(clip_model, images, texts)

    # Setup optimizer with higher learning rate
    optimizer = optax.adamw(learning_rate=1e-4)
    opt_state = optimizer.init(eqx.filter(process_transformer, eqx.is_inexact_array))

    # Train for multiple steps
    losses = []
    for _ in range(50):
        process_transformer, opt_state, loss, _ = step_process_transformer(
            process_transformer,
            img_features,
            text_features,
            states,
            subtasks,
            lengths,
            dense_schemas,
            progress_targets,
            optimizer,
            opt_state,
        )
        losses.append(float(loss))
        print(f"Loss: {loss}")

    # Check that loss decreased
    assert (
        losses[-1] < losses[0]
    ), f"Loss should decrease during training. Initial: {losses[0]:.4f}, Final: {losses[-1]:.4f}"
    assert all(jnp.isfinite(l) for l in losses), "All losses should be finite"


def test_stage_transformer_overfitting(sarm_modules):
    """Test that StageTransformer can overfit to a small batch (sanity check)."""
    _, stage_transformer, clip_model = sarm_modules

    # Create a small fixed batch
    dataset = DummyDataset(timesteps=8, num_cameras=1, size=2)
    batch_data = [dataset[0], dataset[0]]  # Same sample twice

    batch = {
        "img": np.stack([d["img"] for d in batch_data]),
        "text": np.stack([d["text"] for d in batch_data]),
        "state": np.stack([d["state"] for d in batch_data]),
        "subtask": np.stack([d["subtask"] for d in batch_data]),
        "dense_schema": np.stack([d["dense_schema"] for d in batch_data]),
        "length": np.array([d["length"] for d in batch_data]),
    }

    # Prepare batch
    images = jnp.array(batch["img"])
    texts = jnp.array(batch["text"])
    states = jnp.array(batch["state"])
    subtasks = jnp.array(batch["subtask"])
    dense_schemas = jnp.array(batch["dense_schema"])
    lengths = jnp.array(batch["length"])

    subtask_labels = jnp.argmax(subtasks, axis=-1).reshape(-1).astype(jnp.int32)

    # Extract features once
    img_features, text_features = clip_inference(clip_model, images, texts)

    # Setup optimizer
    optimizer = optax.adamw(learning_rate=1e-4)
    opt_state = optimizer.init(eqx.filter(stage_transformer, eqx.is_inexact_array))

    # Train for multiple steps
    losses = []
    accuracies = []
    for _ in range(50):
        stage_transformer, opt_state, loss, _, logits = step_stage_transformer(
            stage_transformer,
            img_features,
            text_features,
            states,
            subtask_labels,
            lengths,
            dense_schemas,
            optimizer,
            opt_state,
        )
        losses.append(float(loss))
        print(f"Loss: {loss}")

        # Convert to labels
        subtask = jnp.argmax(subtasks, axis=-1)
        # Calculate accuracy
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == subtask)
        accuracies.append(float(accuracy))

    # Check that loss decreased
    assert (
        losses[-1] < losses[0]
    ), f"Loss should decrease. Initial: {losses[0]:.4f}, Final: {losses[-1]:.4f}"
    assert all(jnp.isfinite(l) for l in losses), "All losses should be finite"

    # For overfitting on 2 identical samples, we expect high accuracy eventually
    assert accuracies[-1] >= 0.5, f"Accuracy should improve. Final accuracy: {accuracies[-1]:.2f}"
