import numpy as np
import logging
import random
from pathlib import Path
from typing import List, Tuple

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def get_valid_episodes(repo_id: str, root: str | Path | None = None) -> List[int]:
    """
    Collects valid episode indices under the lerobot cache for the given repo_id.

    Args:
        repo_id (str): HuggingFace repo ID,

    Returns:
        List[int]: Sorted list of valid episode indices (e.g., [0, 1, 5, 7, ...])
    """
    dataset = LeRobotDataset(repo_id=repo_id, root=root)
    rewards = dataset.hf_dataset["next.reward"]
    NUM_SUBTASKS = np.ceil(np.max(rewards))
    episodes = list(dataset.meta.episodes["episode_index"])

    non_valid_episodes = []
    for ep_idx in episodes:
        ep = dataset.meta.episodes[ep_idx]
        ep_end = ep["dataset_to_index"]
        if np.ceil(rewards[ep_end - 1]) != NUM_SUBTASKS:
            non_valid_episodes.append(ep_idx)

    if len(non_valid_episodes) > 0:
        logging.warn(
            f"Total Episodes {len(episodes)}. INVALID EPISODES found. Episodes with missing subtasks {non_valid_episodes}"
        )
    valid_episodes = [e for e in episodes if e not in non_valid_episodes]
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
