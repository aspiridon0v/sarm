"""
Script to compute normalization statistics from a LeRobot dataset.

This script loads a LeRobot dataset, extracts state and action data,
uses the SingleFieldLinearNormalizer to compute statistics, and saves
them to a JSON file for later use during training.

Usage:
    # If dataset is in HuggingFace cache (most common):
    python -m sarm.scripts.compute_norm_stats \
        --repo_id <dataset_name> \
        --output_path <path_to_save_json>
    
    # If dataset is in a custom location:
    python -m sarm.scripts.compute_norm_stats \
        --repo_id <dataset_name> \
        --output_path <path_to_save_json> \
        --root <custom_data_root>
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

from sarm.dataset.normalizer import SingleFieldLinearNormalizer


def compute_norm_stats(
    repo_id: str,
    output_path: str,
    root: str | None = None,
    mode: str = "gaussian",
    state_dim: int | None = None,
    action_dim: int | None = None,
    max_frames: int | None = None,
):
    """
    Compute normalization statistics for a LeRobot dataset.

    Args:
        repo_id: HuggingFace repo ID or local path to the dataset
        output_path: Path where to save the JSON file with statistics
        root: Optional root directory for the dataset
        mode: Normalization mode ("limits" or "gaussian")
        state_dim: Optional dimension to slice state data (e.g., 14 for both arms)
        action_dim: Optional dimension to slice action data
        max_frames: Number of frames for the stats computation
    """
    print(f"Loading dataset from {repo_id}...")
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=root,
        download_videos=False,  # We don't need videos for computing stats
    )
    N = len(dataset)
    print(f"Dataset loaded. Total frames: {N}")
    if max_frames is not None and max_frames < N:
        N = max_frames
        print(f"Using {N} frames for calculations")

    print(f"Features: {list(dataset.hf_dataset.features.keys())}")

    # Collect state and action data
    print("Collecting state data...")
    states = []
    for i in tqdm(range(N), desc="Loading states"):
        item = dataset.hf_dataset[i]
        if "observation.state" in item:
            state = item["observation.state"]
            if isinstance(state, torch.Tensor):
                state = state.numpy()
            states.append(state)

    states = np.array(states)
    print(f"State shape: {states.shape}")

    # Apply dimension slicing if specified
    if state_dim is not None and states.shape[-1] > state_dim:
        print(f"Slicing state to first {state_dim} dimensions")
        states = states[..., :state_dim]

    print("Collecting action data...")
    actions = []
    for i in tqdm(range(N), desc="Loading actions"):
        item = dataset.hf_dataset[i]
        if "action" in item:
            action = item["action"]
            if isinstance(action, torch.Tensor):
                action = action.numpy()
            actions.append(action)

    actions = np.array(actions)
    print(f"Action shape: {actions.shape}")

    # Apply dimension slicing if specified
    if action_dim is not None and actions.shape[-1] > action_dim:
        print(f"Slicing action to first {action_dim} dimensions")
        actions = actions[..., :action_dim]

    # Compute normalization statistics
    print(f"\nComputing normalization statistics (mode={mode})...")

    # Fit state normalizer
    print("Fitting state normalizer...")
    state_normalizer = SingleFieldLinearNormalizer.create_fit(
        states,
        last_n_dims=1,
        dtype=torch.float32,
        mode=mode,
    )
    state_stats = state_normalizer.get_input_stats()

    # Fit action normalizer
    print("Fitting action normalizer...")
    action_normalizer = SingleFieldLinearNormalizer.create_fit(
        actions,
        last_n_dims=1,
        dtype=torch.float32,
        mode=mode,
    )
    action_stats = action_normalizer.get_input_stats()

    # Compute quantiles for robust min/max (q01 and q99)
    print("Computing quantiles...")
    state_q01 = np.percentile(states, 1, axis=0).tolist()
    state_q99 = np.percentile(states, 99, axis=0).tolist()
    action_q01 = np.percentile(actions, 1, axis=0).tolist()
    action_q99 = np.percentile(actions, 99, axis=0).tolist()

    # Prepare output dictionary
    norm_stats = {
        "norm_stats": {
            "state": {
                "mean": state_stats["mean"].cpu().numpy().tolist(),
                "std": state_stats["std"].cpu().numpy().tolist(),
                "min": state_stats["min"].cpu().numpy().tolist(),
                "max": state_stats["max"].cpu().numpy().tolist(),
                "q01": state_q01,
                "q99": state_q99,
            },
            "action": {
                "mean": action_stats["mean"].cpu().numpy().tolist(),
                "std": action_stats["std"].cpu().numpy().tolist(),
                "min": action_stats["min"].cpu().numpy().tolist(),
                "max": action_stats["max"].cpu().numpy().tolist(),
                "q01": action_q01,
                "q99": action_q99,
            },
        },
        "metadata": {
            "repo_id": repo_id,
            "num_samples": N,
            "state_shape": list(states.shape),
            "action_shape": list(actions.shape),
            "mode": mode,
        },
    }

    # Save to JSON
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(norm_stats, f, indent=2)

    print(f"\n✓ Normalization statistics saved to: {output_path}")
    print(f"\nState statistics:")
    print(f"  Mean: {state_stats['mean'].cpu().numpy()}")
    print(f"  Std: {state_stats['std'].cpu().numpy()}")
    print(f"\nAction statistics:")
    print(f"  Mean: {action_stats['mean'].cpu().numpy()}")
    print(f"  Std: {action_stats['std'].cpu().numpy()}")


def main():
    parser = argparse.ArgumentParser(description="Compute normalization statistics for a LeRobot dataset")
    parser.add_argument(
        "--repo_id",
        type=str,
        required=True,
        help="HuggingFace repo ID or local path to the dataset",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where to save the JSON file with statistics",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory for the dataset (optional, only needed for custom locations outside HF cache)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["limits", "gaussian"],
        default="gaussian",
        help="Normalization mode (default: gaussian)",
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=None,
        help="Optional: slice state to first N dimensions (e.g., 14 for both arms)",
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=None,
        help="Optional: slice action to first N dimensions",
    )

    args = parser.parse_args()

    compute_norm_stats(
        repo_id=args.repo_id,
        output_path=args.output_path,
        root=args.root,
        mode=args.mode,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
    )


if __name__ == "__main__":
    main()
