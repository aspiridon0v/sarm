import numpy as np

from sarm.scripts.label_dataset import DatasetLabeler

TASK_NAMES = [
    "Grasp right corner",
    "Grasp left corner",
    "Fold towel horizontally",
    "Grasp right edge",
    "Fold towel vertically",
]


def test_reward_computation():
    """Test the reward computation with dummy data."""
    print("\n" + "=" * 50)
    print("Testing Reward Computation (NEW SEMANTICS)")
    print("=" * 50)
    print("\nNEW: Boundaries mark the END of a subtask")
    print("- First subtask starts at episode beginning")
    print("- Last subtask ends at episode end")

    # Create a mock labeler for testing
    class MockMeta:
        def __init__(self):
            self.episodes = [
                {"dataset_from_index": 0, "dataset_to_index": 100},
                {"dataset_from_index": 100, "dataset_to_index": 200},
                {"dataset_from_index": 200, "dataset_to_index": 300},
                {"dataset_from_index": 300, "dataset_to_index": 400},
            ]

    class MockDataset:
        def __init__(self):
            self.meta = MockMeta()

        def __len__(self):
            return 400

    labeler = DatasetLabeler.__new__(DatasetLabeler)
    labeler.dataset = MockDataset()
    labeler.num_episodes = 4
    labeler.num_subtasks = len(TASK_NAMES)

    # Add some test boundaries (remember: these mark the END of subtasks)
    labeler.boundaries = {
        0: [
            (20, 0),
            (50, 1),
            (80, 2),
        ],  # Episode 0: subtask 0 ends at 20, 1 ends at 50, 2 ends at 80
        1: [(120, 0), (160, 1)],  # Episode 1: subtask 0 ends at 120, 1 ends at 160
        2: [],  # Episode 2: no boundaries (all subtask 0)
        3: [(320, 0), (340, 1), (360, 2), (380, 3)],  # Episode 3: 4 boundaries
    }

    # Compute rewards
    rewards = labeler.compute_rewards()

    # Print some examples
    print(f"\nTotal frames: {len(rewards)}")
    print(f"\nExample rewards (Episode 0: 0-99):")
    print(f"  Frame 0 (start, subtask 0):       {rewards[0]:.3f}")
    print(f"  Frame 10 (middle of subtask 0):   {rewards[10]:.3f}")
    print(f"  Frame 20 (END of subtask 0):      {rewards[20]:.3f}")
    print(f"  Frame 21 (start of subtask 1):    {rewards[21]:.3f}")
    print(f"  Frame 35 (middle of subtask 1):   {rewards[35]:.3f}")
    print(f"  Frame 50 (END of subtask 1):      {rewards[50]:.3f}")
    print(f"  Frame 51 (start of subtask 2):    {rewards[51]:.3f}")
    print(f"  Frame 80 (END of subtask 2):      {rewards[80]:.3f}")
    print(f"  Frame 81 (start of subtask 3):    {rewards[81]:.3f}")
    print(f"  Frame 99 (end of episode):        {rewards[99]:.3f}")
    print(f"\nEpisode 2 (200-299, no labels):")
    print(f"  Frame 200 (all subtask 0):        {rewards[200]:.3f}")
    print(f"  Frame 250 (all subtask 0):        {rewards[250]:.3f}")

    # Verify monotonicity within episodes
    for ep in range(labeler.num_episodes):
        start_idx, end_idx = labeler.get_episode_frames(ep)

        ep_rewards = rewards[start_idx:end_idx]
        if len(ep_rewards) > 0 and not np.all(ep_rewards == 0):
            # Check if rewards are non-decreasing
            is_monotonic = np.all(np.diff(ep_rewards) >= 0)
            print(
                f"\nEpisode {ep}: rewards are {'monotonically increasing ✓' if is_monotonic else 'NOT monotonic ✗'}"
            )

    print("\n" + "=" * 50)
