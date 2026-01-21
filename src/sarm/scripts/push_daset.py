from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Option A: if you have the dataset directory locally
output_path = Path(__file__).parent / "data" / "towel_base_with_rewards"

dataset = LeRobotDataset(repo_id="ETHRC/towel_base_with_rewards", root=output_path)

dataset.push_to_hub(repo_id="ETHRC/towel_base_with_rewards")
