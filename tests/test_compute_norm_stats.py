"""
Test the compute_norm_stats script.
"""

import json
from pathlib import Path

import pytest
import jax.numpy as jnp

from sarm.dataset.normalizer import _load_normalized_sarm_reward, normalize_reward
from sarm.scripts.compute_norm_stats import compute_norm_stats

@pytest.fixture
def mock_data_file(tmp_path):
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
        },
        "sarm_rewards": {
            "0.0": 0.1850759809480608,
            "1.0": 0.2585198352207694,
            "2.0": 0.4776013123262671,
            "3.0": 0.5404853708323883,
            "4.0": 1.0
        }
    }
    output_path = Path(tmp_path) / "norm_stats.json"
    with open(output_path, "w") as f:
        json.dump(mock_data, f)
    return output_path

def test_compute_norm_stats_integration(tmp_path):
    """
    Integration test that computes normalization stats from a real dataset.
    This test is skipped by default as it requires downloading data.

    To run manually:
        pytest tests/test_compute_norm_stats.py::test_compute_norm_stats_integration -v -s
    """
    output_path = Path(tmp_path) / "norm_stats.json"

    # Use a small test dataset or the actual dataset
    repo_id = "ETHRC/towel_base_with_rewards"

    compute_norm_stats(repo_id=repo_id,
                       output_path=str(output_path),
                       mode="gaussian",
                       max_frames=100,
                       with_rewards=True)

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

    assert 'sarm_rewards' in data

    print("✓ All checks passed!")


def test_normalizer_loading_format(tmp_path, mock_data_file):
    """
    Test that the JSON format matches what get_normalizer_from_calculated expects.
    """
    # Create a mock JSON structure


    import torch
    from sarm.dataset.normalizer import get_normalizer_from_calculated
    normalizer = get_normalizer_from_calculated(mock_data_file, device="cpu")

    assert normalizer is not None
    test_data = torch.randn(10, 14)
    normalized = normalizer.normalize(test_data)
    assert normalized.shape == test_data.shape
    unnormalized = normalizer.unnormalize(normalized)
    assert torch.allclose(unnormalized, test_data, rtol=1e-4)

def test_get_reward_normalizer(mock_data_file):
    norm_values = _load_normalized_sarm_reward(mock_data_file)
    assert jnp.array_equal(norm_values, jnp.array([0.1850759809480608, 0.2585198352207694, 0.4776013123262671, 0.5404853708323883, 1.0]))

    sub_tasks = len(norm_values)
    v_1 = jnp.array(0)
    r_1 = normalize_reward(v_1, norm_values)
    assert r_1 == 0
    v_2 = jnp.array(1.2)
    r_2 = normalize_reward(v_2, norm_values)
    assert r_2 > norm_values[0] and r_2 < norm_values[1]
    v_3 = jnp.array(sub_tasks + 0.99999999999)
    r_3 = normalize_reward(v_3, norm_values)
    assert r_3 > norm_values[-2]
    assert r_3 <= 1
    assert jnp.allclose(r_3, 1, atol=1e-4)

    v_4 = jnp.array(4.1)
    r_4 = normalize_reward(v_4, norm_values)
    assert r_4 > norm_values[-2] and r_4 < norm_values[-1]

    vec = jnp.array([v_1, v_2, v_3, v_4])
    vec_r = normalize_reward(vec, norm_values)
    assert jnp.array_equal(vec_r, jnp.array([r_1, r_2, r_3, r_4]))

    batch = jnp.array([[v_1, v_2, v_3, v_4],
                       [v_4, v_3, v_2, v_1],
                       ])
    batch_r = normalize_reward(batch, norm_values)
    assert jnp.array_equal(batch_r, jnp.array([[r_1, r_2, r_3, r_4],
                                             [r_4, r_3, r_2, r_1],]
                                            ))

if __name__ == "__main__":
    # Run the format test
    test_normalizer_loading_format()
    print("\nTo run the full integration test (requires dataset download):")
    print("  pytest tests/test_compute_norm_stats.py::test_compute_norm_stats_integration -v -s")
