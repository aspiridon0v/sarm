import random
import re
from pathlib import Path
from typing import List, Tuple


def get_valid_episodes(repo_id: str, root: str | Path | None = None) -> List[int]:
    """
    Collects valid episode indices under the lerobot cache for the given repo_id.

    Args:
        repo_id (str): HuggingFace repo ID,

    Returns:
        List[int]: Sorted list of valid episode indices (e.g., [0, 1, 5, 7, ...])
    """
    if root is None:
        base_path = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id / "data"
    else:
        base_path = Path(root)

    episode_pattern = re.compile(r"episode_(\d+)\.parquet")

    valid_episodes = []

    if not base_path.exists():
        raise FileNotFoundError(f"Data directory not found: {base_path}")

    for chunk_dir in base_path.glob("chunk-*"):
        if not chunk_dir.is_dir():
            continue
        for file in chunk_dir.glob("episode_*.parquet"):
            match = episode_pattern.match(file.name)
            if match:
                ep_idx = int(match.group(1))
                valid_episodes.append(ep_idx)

    return sorted(valid_episodes)


def split_train_eval_episodes(
    valid_episodes: List[int], train_ratio: float = 0.9, seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Randomly split valid episodes into training and evaluation sets.

    Args:
        valid_episodes (List[int]): List of valid episode indices.
        train_ratio (float): Fraction of episodes to use for training (default: 0.9).
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        Tuple[List[int], List[int]]: (train_episodes, eval_episodes)
    """
    random.seed(seed)
    episodes = valid_episodes.copy()
    random.shuffle(episodes)

    split_index = int(len(episodes) * train_ratio)
    train_episodes = episodes[:split_index]
    eval_episodes = episodes[split_index:]

    return train_episodes, eval_episodes