import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata, LeRobotDataset

from sarm.dataset.gap_dataset import GapLerobotDataset


def test_gap_dataset():
    repo_id = "ETHRC/towel_base_with_rewards"

    dataset_meta = LeRobotDatasetMetadata(repo_id)
    action_horizon = 25
    dataset = LeRobotDataset(
        repo_id,
        delta_timestamps={key: [t / dataset_meta.fps for t in range(action_horizon)] for key in ("action",)},
    )
    # Test Action Chunk
    dataset_gap = GapLerobotDataset(repo_id=repo_id, action_horizon=25, frame_gap=30, t_step_lookback=8)
    np.testing.assert_array_equal(dataset[124]["action"][0], dataset[100]["action"][24])
    assert dataset[0]["action"].shape == (action_horizon, dataset_meta.shapes["action"][0])
    np.testing.assert_array_equal(dataset_gap[100]["action"], dataset[100]["action"])

    # Test Gap Data
    assert "gap_data_0.action" in dataset_gap[0]
    assert "gap_data_1.action" in dataset_gap[0]


def test_gap_dataset_with_non_contiguous_episodes():
    """
    Test that GapLerobotDataset works correctly with non-contiguous episodes.

    When using the episodes parameter with non-contiguous episode indices (e.g., [5, 10]),
    the dataset should correctly handle the index mapping between the filtered dataset
    and the absolute episode indices.
    """
    repo_id = "ETHRC/towel_base_with_rewards"

    # Use non-contiguous episodes to trigger the bug
    episodes = [5, 10]

    dataset_gap = GapLerobotDataset(
        repo_id=repo_id,
        action_horizon=25,
        frame_gap=30,
        t_step_lookback=8,
        episodes=episodes,
    )

    # This should not raise an IndexError
    # Currently fails because _get_hist_data uses absolute indices
    # but hf_dataset is filtered and uses relative indices
    item = dataset_gap[0]

    # Verify the item has expected keys
    assert "gap_data_0.action" in item
    assert "gap_data_1.action" in item

    # Try accessing an item from the second episode as well
    # Find the start of second episode in the filtered dataset
    ep_5_length = dataset_gap.meta.episodes[5]["dataset_to_index"] - dataset_gap.meta.episodes[5]["dataset_from_index"]
    item_from_ep_10 = dataset_gap[ep_5_length + 1]
    assert "gap_data_0.action" in item_from_ep_10
    assert "gap_data_1.action" in item_from_ep_10
