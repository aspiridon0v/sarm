"""
Test the compute_norm_stats script.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sarm.scripts.compute_norm_stats import compute_norm_stats


def test_compute_norm_stats_integration():
    """
    Integration test that computes normalization stats from a real dataset.
    This test is skipped by default as it requires downloading data.

    To run manually:
        pytest tests/test_compute_norm_stats.py::test_compute_norm_stats_integration -v -s
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "norm_stats.json"

        # Use a small test dataset or the actual dataset
        repo_id = "ETHRC/piper_towel_v0_with_rewards"

        compute_norm_stats(
            repo_id=repo_id,
            output_path=str(output_path),
            mode="gaussian",
        )

        # Verify the file was created
        assert output_path.exists()

        # Load and verify structure
        with open(output_path, "r") as f:
            data = json.load(f)

        assert "norm_stats" in data
        assert "state" in data["norm_stats"]
        assert "action" in data["norm_stats"]

        # Check state stats
        state_stats = data["norm_stats"]["state"]
        assert "mean" in state_stats
        assert "std" in state_stats
        assert "min" in state_stats
        assert "max" in state_stats
        assert "q01" in state_stats
        assert "q99" in state_stats

        # Check action stats
        action_stats = data["norm_stats"]["action"]
        assert "mean" in action_stats
        assert "std" in action_stats
        assert "min" in action_stats
        assert "max" in action_stats
        assert "q01" in action_stats
        assert "q99" in action_stats

        # Verify dimensions
        if "metadata" in data:
            print(f"Metadata: {data['metadata']}")

        # Check that state has 14 dimensions (as specified)
        assert len(state_stats["mean"]) == 14
        assert len(state_stats["std"]) == 14

        print("✓ All checks passed!")


def test_normalizer_loading_format():
    """
    Test that the JSON format matches what get_normalizer_from_calculated expects.
    """
    # Create a mock JSON structure
    mock_data = {
        "norm_stats": {
            "state": {
                "mean": [0.0] * 14,
                "std": [1.0] * 14,
                "min": [-1.0] * 14,
                "max": [1.0] * 14,
                "q01": [-0.9] * 14,
                "q99": [0.9] * 14,
            },
            "action": {
                "mean": [0.0] * 14,
                "std": [1.0] * 14,
                "min": [-1.0] * 14,
                "max": [1.0] * 14,
                "q01": [-0.9] * 14,
                "q99": [0.9] * 14,
            },
        }
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mock_data, f)
        temp_path = f.name

    try:
        # Try to load it using the normalizer function
        import torch

        from sarm.dataset.normalizer import get_normalizer_from_calculated

        normalizer = get_normalizer_from_calculated(temp_path, device="cpu")

        # Verify it's a valid normalizer
        assert normalizer is not None

        # Test normalization
        test_data = torch.randn(10, 14)
        normalized = normalizer.normalize(test_data)
        assert normalized.shape == test_data.shape

        # Test unnormalization
        unnormalized = normalizer.unnormalize(normalized)
        assert torch.allclose(unnormalized, test_data, rtol=1e-4)

        print("✓ Normalizer loading and format check passed!")

    finally:
        Path(temp_path).unlink()


if __name__ == "__main__":
    # Run the format test
    test_normalizer_loading_format()
    print("\nTo run the full integration test (requires dataset download):")
    print("  pytest tests/test_compute_norm_stats.py::test_compute_norm_stats_integration -v -s")
