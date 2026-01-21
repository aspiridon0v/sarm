"""
Pytest configuration for controlling CPU/GPU device selection.

Set the TEST_DEVICE environment variable to control where tests run:
- TEST_DEVICE=cpu    : Force CPU for both JAX and PyTorch
- TEST_DEVICE=gpu    : Force GPU for both JAX and PyTorch (requires GPU availability)
- TEST_DEVICE=auto   : Auto-detect (default) - use GPU if available, otherwise CPU

Examples:
    # Run tests on CPU only
    TEST_DEVICE=cpu pytest tests/

    # Run tests on GPU only (will fail if GPU not available)
    TEST_DEVICE=gpu pytest tests/

    # Auto-detect (default behavior)
    pytest tests/
"""

import os
import random
import sys
from pathlib import Path

import jax
import numpy as np
import pytest
import torch
from sarm.utils.convert_clip import main as export_clip_weights

@pytest.fixture(scope="session")
def clip_weight_path():
    path = Path(__file__).parents[1] / 'checkpoints' / 'clip_vit_b32_openai.npz'
    return str(path)

@pytest.fixture(scope="session")
def ensure_weights(clip_weight_path):
    """Export weights once per session if not already present."""
    if not os.path.exists(clip_weight_path):
        export_clip_weights()
    assert os.path.exists(clip_weight_path), "Failed to export CLIP weights to .npz"
    return clip_weight_path

def pytest_configure(config):
    """Configure JAX and PyTorch devices before running tests."""
    # Set seeds for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = os.environ.get("TEST_DEVICE", "auto").lower()

    if device not in ["cpu", "gpu", "auto"]:
        raise ValueError(f"Invalid TEST_DEVICE='{device}'. Must be 'cpu', 'gpu', or 'auto'.")

    # Configure JAX
    if device == "cpu":
        # Force JAX to use CPU
        jax.config.update("jax_platform_name", "cpu")
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"TEST DEVICE: CPU (forced)", file=sys.stderr)
        print(f"{'='*60}\n", file=sys.stderr)
    elif device == "gpu":
        # Force JAX to use GPU
        jax.config.update("jax_platform_name", "gpu")
        # Check if GPU is available
        try:
            devices = jax.devices("gpu")
            if len(devices) == 0:
                raise RuntimeError("TEST_DEVICE=gpu but no GPU devices available for JAX")
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"TEST DEVICE: GPU (forced)", file=sys.stderr)
            print(f"JAX GPU devices: {devices}", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)
        except RuntimeError as e:
            raise RuntimeError(f"TEST_DEVICE=gpu but GPU not available: {e}")
    else:  # auto
        # Let JAX choose the default device
        try:
            gpu_devices = jax.devices("gpu")
            if len(gpu_devices) > 0:
                print(f"\n{'='*60}", file=sys.stderr)
                print(f"TEST DEVICE: GPU (auto-detected)", file=sys.stderr)
                print(f"JAX GPU devices: {gpu_devices}", file=sys.stderr)
                print(f"{'='*60}\n", file=sys.stderr)
            else:
                print(f"\n{'='*60}", file=sys.stderr)
                print(
                    f"TEST DEVICE: CPU (auto-detected, no GPU available)",
                    file=sys.stderr,
                )
                print(f"{'='*60}\n", file=sys.stderr)
        except RuntimeError:
            print(f"\n{'='*60}", file=sys.stderr)
            print(f"TEST DEVICE: CPU (auto-detected)", file=sys.stderr)
            print(f"{'='*60}\n", file=sys.stderr)


@pytest.fixture(scope="session")
def torch_device():
    """
    Provide the torch device based on TEST_DEVICE environment variable.

    Returns:
        torch.device: The device to use for PyTorch tensors
    """
    device = os.environ.get("TEST_DEVICE", "auto").lower()

    if device == "cpu":
        return torch.device("cpu")
    elif device == "gpu":
        if not torch.cuda.is_available():
            raise RuntimeError("TEST_DEVICE=gpu but CUDA is not available for PyTorch")
        return torch.device("cuda")
    else:  # auto
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@pytest.fixture(scope="session")
def device_info(torch_device):
    """
    Provide information about the current device configuration.

    Returns:
        dict: Dictionary with device information
    """
    jax_devices = jax.devices()
    jax_default = jax.devices()[0]

    info = {
        "torch_device": torch_device,
        "torch_device_name": str(torch_device),
        "jax_devices": jax_devices,
        "jax_default_device": jax_default,
        "jax_backend": jax_default.platform,
    }

    return info
