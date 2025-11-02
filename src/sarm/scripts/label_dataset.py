#!/usr/bin/env python
"""
Interactive script to label subtask boundaries in a robot dataset.
Displays three camera views and allows keyboard labeling of subtask transitions.

Usage:
    python -m sarm.scripts.label_dataset

Features:
    - Displays three camera views side-by-side (wrist1, wrist2, stereo)
    - Navigate through episodes and frames using keyboard controls
    - Mark subtask boundaries by pressing number keys (1-5)
    - Automatically computes reward as: subtask_id + linear_progress
      where linear_progress ∈ [0, 1) represents progress to next boundary
    - Saves rewards.npy and rewards_boundaries.npy files

Keyboard Controls:
    SPACE       : Play/Pause video playback
    LEFT/RIGHT  : Navigate frame by frame
    N/P         : Next/Previous episode
    1-5         : Mark END of subtask (1=end of "Grasp right corner", etc.)
    U           : Undo last boundary mark
    S           : Save rewards and quit
    Q           : Quit without saving

Labeling Logic:
    - First subtask (0) starts AUTOMATICALLY at episode beginning
    - Pressing 1-5 marks the END of that subtask at the current frame
    - The next subtask starts at frame+1 after the boundary
    - Last subtask ends AUTOMATICALLY at episode end (no boundary needed)

Reward Computation Example:
    If you mark END of subtasks at frames [100, 200, 300] for subtasks [0, 1, 2]:
    - Frames 0-100:   Subtask 0, reward = 0 + progress (0.0 to 0.99)
    - Frames 101-200: Subtask 1, reward = 1 + progress (1.0 to 1.99)
    - Frames 201-300: Subtask 2, reward = 2 + progress (2.0 to 2.99)
    - Frames 301+:    Subtask 3, reward = 3 + progress (3.0+)

Output Files:
    When you press 'S' to save, the script creates:
    1. data/[dataset_name]_with_rewards/: Complete dataset with 'next.reward' feature
       - Full LeRobotDataset format (loadable with LeRobotDataset(root=...))
       - All original features plus new 'next.reward' feature
       - labeling_metadata.txt: Human-readable labeling info
       - Compatible with LeRobot's visualize_dataset.py script
       - FAST: Videos are copied directly without re-encoding (huge time savings!)
"""

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm

TASK_NAMES = [
    "Grasp right corner",
    "Grasp left corner",
    "Fold towel horizontally",
    "Grasp right edge",
    "Fold towel vertically",
]

# Camera keys in the dataset
CAMERA_KEYS = [
    "observation.images.wrist1",
    "observation.images.wrist2",
    "observation.images.stereo",
]


class DatasetLabeler:
    """Interactive labeler for robot datasets with subtask boundary annotation."""

    def __init__(self, repo_id: str, task_names: List[str]):
        self.dataset = LeRobotDataset(repo_id=repo_id)
        self.task_names = task_names
        self.num_subtasks = len(task_names)

        # Get episode information from lerobot dataset
        self.num_episodes = self.dataset.num_episodes
        self.episode_starts = self.dataset.episode_data_index["from"].tolist()
        self.episode_ends = self.dataset.episode_data_index["to"].tolist()

        print(f"Dataset loaded: {repo_id}")
        print(f"Number of episodes: {self.num_episodes}")
        print(f"Total frames: {len(self.dataset)}")

        # Labeling state
        self.current_episode = 0
        self.current_frame = 0
        self.paused = True
        self.playback_speed = 1  # frames per update

        # Subtask boundaries: dict[episode_idx] -> list of (frame_idx, subtask_idx)
        self.boundaries: Dict[int, List[tuple]] = {ep: [] for ep in range(self.num_episodes)}

        # Running state
        self.running = True

        # Window name
        self.window_name = "Dataset Labeler"

    def get_episode_frames(self, episode_idx: int):
        """Get start and end frame indices for an episode."""
        start_idx = self.episode_starts[episode_idx]
        end_idx = self.episode_ends[episode_idx]
        return start_idx, end_idx

    def tensor_to_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a torch tensor image to numpy array for opencv."""
        # Tensor is in range [0, 1] with shape (C, H, W) or (H, W, C)
        if tensor.dim() == 3 and tensor.shape[0] in [1, 3, 4]:
            # CHW format -> HWC
            img = tensor.permute(1, 2, 0).cpu().numpy()
        else:
            img = tensor.cpu().numpy()

        # Convert to uint8
        img = (img * 255).astype(np.uint8)

        # Convert RGB to BGR for opencv
        if img.shape[-1] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return img

    def create_display_frame(self, frame_data: Dict) -> np.ndarray:
        """Create a combined display with all three camera views."""
        images = []
        for cam_key in CAMERA_KEYS:
            if cam_key in frame_data:
                img = self.tensor_to_image(frame_data[cam_key])
                # Resize for display (make smaller if needed)
                img = cv2.resize(img, (320, 240))
                # Add camera name label with black background for better visibility
                cam_name = cam_key.split(".")[-1]
                text_size = cv2.getTextSize(cam_name, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(img, (5, 5), (15 + text_size[0], 35), (0, 0, 0), -1)
                cv2.putText(img, cam_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                images.append(img)

        # Stack images horizontally
        camera_view = np.hstack(images)

        # Create info panel at top (for episode info and boundaries)
        info_panel_height = 150
        info_panel = np.zeros((info_panel_height, camera_view.shape[1], 3), dtype=np.uint8)

        # Add episode and frame info to top panel
        start_idx, end_idx = self.get_episode_frames(self.current_episode)
        frame_in_episode = self.current_frame - start_idx
        total_frames_in_episode = end_idx - start_idx

        episode_info = [
            f"Episode: {self.current_episode}/{self.num_episodes-1}  |  Frame: {frame_in_episode}/{total_frames_in_episode-1} (Global: {self.current_frame})  |  Status: {'PAUSED' if self.paused else 'PLAYING'}",
        ]

        for i, text in enumerate(episode_info):
            cv2.putText(
                info_panel,
                text,
                (10, 20 + i * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

        # Show boundaries for current episode in the top panel
        if self.current_episode in self.boundaries and self.boundaries[self.current_episode]:
            boundary_text = f"Marked boundaries:"
            cv2.putText(
                info_panel, boundary_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            for i, (frame_idx, subtask_idx) in enumerate(self.boundaries[self.current_episode]):
                start_idx, _ = self.get_episode_frames(self.current_episode)
                frame_in_ep = frame_idx - start_idx
                boundary_info = (
                    f"  {i+1}. Frame {frame_in_ep}: END '{self.task_names[subtask_idx]}'"
                )
                cv2.putText(
                    info_panel,
                    boundary_info,
                    (10, 80 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )
        else:
            cv2.putText(
                info_panel,
                "No boundaries marked yet",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (128, 128, 128),
                1,
            )

        # Create controls panel at bottom
        controls_panel_height = 200
        controls_panel = np.zeros((controls_panel_height, camera_view.shape[1], 3), dtype=np.uint8)

        controls_text = [
            "Controls:",
            "  SPACE: Play/Pause",
            "  LEFT/RIGHT: Previous/Next frame",
            "  N: Next episode  |  P: Previous episode",
            f"  1-{self.num_subtasks}: Mark END of subtask",
            "  U: Undo last boundary  |  S: Save dataset  |  Q: Quit",
            "",
            "NOTE: 1st subtask starts automatically at frame 0",
        ]

        for i, text in enumerate(controls_text):
            cv2.putText(
                controls_panel,
                text,
                (10, 25 + i * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        # Stack everything vertically: info panel, camera views, controls panel
        combined = np.vstack([info_panel, camera_view, controls_panel])

        return combined

    def move_frame(self, delta: int):
        """Move to a different frame."""
        start_idx, end_idx = self.get_episode_frames(self.current_episode)
        new_frame = self.current_frame + delta
        self.current_frame = max(start_idx, min(end_idx - 1, new_frame))

    def next_episode(self):
        """Move to the next episode."""
        if self.current_episode < self.num_episodes - 1:
            self.current_episode += 1
            start_idx, _ = self.get_episode_frames(self.current_episode)
            self.current_frame = start_idx

    def prev_episode(self):
        """Move to the previous episode."""
        if self.current_episode > 0:
            self.current_episode -= 1
            start_idx, _ = self.get_episode_frames(self.current_episode)
            self.current_frame = start_idx

    def mark_boundary(self, subtask_idx: int):
        """Mark a subtask boundary at the current frame.

        This marks the END of subtask_idx (and implicitly the start of subtask_idx+1).
        """
        if 0 <= subtask_idx < self.num_subtasks:
            self.boundaries[self.current_episode].append((self.current_frame, subtask_idx))
            # Sort boundaries by frame index
            self.boundaries[self.current_episode].sort(key=lambda x: x[0])
            print(
                f"Marked END of '{self.task_names[subtask_idx]}' at frame {self.current_frame} "
                f"(Episode {self.current_episode})"
            )
            if subtask_idx + 1 < self.num_subtasks:
                print(
                    f"  → Next subtask '{self.task_names[subtask_idx + 1]}' starts at frame {self.current_frame + 1}"
                )

    def undo_last_boundary(self):
        """Remove the last boundary in the current episode."""
        if self.current_episode in self.boundaries and self.boundaries[self.current_episode]:
            removed = self.boundaries[self.current_episode].pop()
            print(f"Removed boundary: {removed}")

    def compute_rewards(self) -> np.ndarray:
        """
        Compute reward for each frame as: subtask_id + linear_progress
        where linear_progress is normalized distance to next boundary [0, 1).

        A boundary (frame_idx, subtask_id) means "subtask_id ENDS at frame_idx".
        - First subtask (0) starts automatically at episode start
        - After boundary marking end of subtask_id, subtask_id+1 starts at frame_idx+1
        - Last subtask automatically ends at episode end (no boundary needed)
        """
        rewards = np.zeros(len(self.dataset), dtype=np.float32)

        for episode_idx in range(self.num_episodes):
            start_idx, end_idx = self.get_episode_frames(episode_idx)
            boundaries = self.boundaries.get(episode_idx, [])

            # Sort boundaries by frame index (should already be sorted, but just in case)
            boundaries = sorted(boundaries, key=lambda x: x[0])

            if not boundaries:
                # No boundaries labeled: entire episode is subtask 0
                segment_length = end_idx - start_idx
                for frame_idx in range(start_idx, end_idx):
                    if segment_length > 0:
                        progress = (frame_idx - start_idx) / segment_length
                    else:
                        progress = 0.0
                    rewards[frame_idx] = 0.0 + progress
                continue

            # Build segments: [(start_frame, end_frame, subtask_id), ...]
            segments = []

            # First segment: episode start to first boundary (inclusive)
            segments.append((start_idx, boundaries[0][0], 0))

            # Middle segments: after each boundary to the next (or episode end)
            for i in range(len(boundaries)):
                boundary_frame, ended_subtask_id = boundaries[i]
                next_subtask_id = ended_subtask_id + 1

                # Start frame is the frame after the boundary
                segment_start = boundary_frame + 1

                # End frame is either the next boundary or episode end
                if i < len(boundaries) - 1:
                    segment_end = boundaries[i + 1][0]
                else:
                    segment_end = end_idx - 1  # inclusive

                # Only add segment if there are frames in it
                if segment_start <= segment_end:
                    segments.append((segment_start, segment_end, next_subtask_id))

            # Compute rewards for each segment
            for segment_start, segment_end, subtask_id in segments:
                # Include segment_end in the range (it's inclusive)
                segment_length = segment_end - segment_start + 1
                for frame_idx in range(segment_start, segment_end + 1):
                    if segment_length > 0:
                        progress = (frame_idx - segment_start) / segment_length
                    else:
                        progress = 0.0
                    rewards[frame_idx] = subtask_id + progress

        return rewards

    def save_dataset_with_rewards_fast(self, output_dir: str = "data"):
        """Save the dataset with 'next.reward' by copying videos and modifying parquet files only.

        This is much faster than save_dataset_with_rewards() since it avoids re-encoding videos.

        Args:
            output_dir: Directory to save the new dataset (default: "data")
        """
        import shutil
        from pathlib import Path

        import pyarrow as pa
        import pyarrow.parquet as pq

        # Compute rewards
        print("\n  Computing rewards...")
        rewards = self.compute_rewards()
        print(f"  ✓ Rewards computed for {len(rewards)} frames")

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 50)
        print("Fast Save: Copying videos and modifying parquet files...")
        print("=" * 50)

        # Dataset name
        dataset_name = self.dataset.repo_id.split("/")[-1] + "_with_rewards"
        full_output_path = output_path / dataset_name

        print(f"\nCreating new LeRobotDataset: {dataset_name}")
        print(f"  Output: {full_output_path}")
        print(f"  Adding 'next.reward' column")
        print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")

        # Remove existing directory if it exists
        if full_output_path.exists():
            print(f"  Removing existing directory...")
            shutil.rmtree(full_output_path)

        # Get source dataset root path
        source_root = Path(self.dataset.root)

        print(f"\n1. Copying directory structure and videos...")

        # Copy videos directory directly (no re-encoding!)
        if (source_root / "videos").exists():
            print(f"  Copying videos/...")
            shutil.copytree(source_root / "videos", full_output_path / "videos")

        # Copy images directory if it exists (for non-video datasets)
        if (source_root / "images").exists():
            print(f"  Copying images/...")
            shutil.copytree(source_root / "images", full_output_path / "images")

        # Copy meta directory
        print(f"  Copying meta/...")
        shutil.copytree(source_root / "meta", full_output_path / "meta")

        # Update info.json to add next.reward feature
        import json

        info_path = full_output_path / "meta" / "info.json"
        with open(info_path, "r") as f:
            info = json.load(f)

        # Add next.reward to features
        info["features"]["next.reward"] = {"dtype": "float32", "shape": [1]}
        info["data_path"] = (
            f"data/chunk-{{episode_chunk:03d}}/episode_{{episode_index:06d}}.parquet"
        )

        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        print(f"\n2. Modifying parquet files to add rewards...")

        # Create data directory structure
        (full_output_path / "data").mkdir(parents=True, exist_ok=True)

        # Process each parquet file
        source_data_dir = source_root / "data"
        for chunk_dir in sorted(source_data_dir.iterdir()):
            if not chunk_dir.is_dir():
                continue

            # Create corresponding output chunk directory
            output_chunk_dir = full_output_path / "data" / chunk_dir.name
            output_chunk_dir.mkdir(parents=True, exist_ok=True)

            # Process each parquet file in the chunk
            for parquet_file in sorted(chunk_dir.glob("*.parquet")):
                # Read the original parquet file
                table = pq.read_table(parquet_file)

                # Determine the frame indices for this episode
                episode_indices = table.column("index").to_pylist()
                start_idx = episode_indices[0]
                end_idx = episode_indices[-1] + 1

                # Get rewards for this episode
                episode_rewards = rewards[start_idx:end_idx]

                # Create next.reward column with shape (N, 1) to match LeRobot format
                reward_column = pa.array(
                    [[r] for r in episode_rewards], type=pa.list_(pa.float32(), 1)
                )

                # Add the new column to the table
                new_table = table.append_column("next.reward", reward_column)

                # Write the modified table to the output directory
                output_file = output_chunk_dir / parquet_file.name
                pq.write_table(new_table, output_file)

                print(f"  ✓ {parquet_file.name}")

        print(f"\n✓ Dataset saved successfully (no video re-encoding needed)!")
        print(f"  Location: {full_output_path}")

        # Save metadata
        metadata_path = full_output_path / "labeling_metadata.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Original dataset: {self.dataset.repo_id}\n")
            f.write(f"Total frames: {len(self.dataset)}\n")
            f.write(f"Number of episodes: {self.num_episodes}\n")
            f.write(f"Reward range: [{rewards.min():.6f}, {rewards.max():.6f}]\n")
            f.write(f"\nSubtasks:\n")
            for i, name in enumerate(self.task_names):
                f.write(f"  {i}. {name}\n")
            f.write(f"\nBoundaries (marking END of subtasks):\n")
            for ep_idx in range(self.num_episodes):
                if self.boundaries.get(ep_idx):
                    f.write(f"  Episode {ep_idx}:\n")
                    for frame_idx, subtask_idx in self.boundaries[ep_idx]:
                        start_idx, _ = self.get_episode_frames(ep_idx)
                        frame_in_ep = frame_idx - start_idx
                        f.write(
                            f"    Frame {frame_in_ep} (global {frame_idx}): END of '{self.task_names[subtask_idx]}'\n"
                        )
                else:
                    f.write(f"  Episode {ep_idx}: No boundaries\n")
        print(f"    └── labeling_metadata.txt")

        print(f"\n💡 Load with:")
        print(f"   from lerobot.datasets.lerobot_dataset import LeRobotDataset")
        print(f'   ds = LeRobotDataset("{dataset_name}", root="{full_output_path}")')

        return full_output_path

    def save_dataset_with_rewards(self, output_dir: str = "data"):
        """Save the dataset with an additional 'next.reward' feature as a proper LeRobotDataset.

        This uses the LeRobotDataset API to create a new dataset with the 'next.reward' feature.
        Can be loaded with: LeRobotDataset(repo_id="dataset_name_with_rewards", root=output_dir)

        NOTE: This method re-encodes videos which is slow. Use save_dataset_with_rewards_fast() instead
        if you want to avoid re-encoding (much faster for large datasets).

        Args:
            output_dir: Directory to save the new dataset (default: "data")
        """
        import shutil
        from pathlib import Path

        # Compute rewards
        print("\n  Computing rewards...")
        rewards = self.compute_rewards()
        print(f"  ✓ Rewards computed for {len(rewards)} frames")

        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 50)
        print("Saving LeRobotDataset with rewards...")
        print("=" * 50)

        # Dataset name
        dataset_name = self.dataset.repo_id.split("/")[-1] + "_with_rewards"
        full_output_path = output_path / dataset_name

        print(f"\nCreating new LeRobotDataset: {dataset_name}")
        print(f"  Output: {full_output_path}")
        print(f"  Adding 'next.reward' column with shape: {rewards.shape}")
        print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")

        # Remove existing directory if it exists
        if full_output_path.exists():
            print(f"  Removing existing directory...")
            shutil.rmtree(full_output_path)

        # Add reward feature to the existing features (use "next.reward" for LeRobot visualization compatibility)
        new_features = {**self.dataset.features, "next.reward": {"dtype": "float32", "shape": (1,)}}

        # Create new dataset using LeRobot API
        print(f"\n  Creating dataset with LeRobot API...")
        new_dataset = LeRobotDataset.create(
            repo_id=dataset_name,
            fps=self.dataset.fps,
            features=new_features,
            root=full_output_path,  # Pass the full path, not parent
            robot_type=self.dataset.meta.robot_type,
            use_videos=len(self.dataset.meta.video_keys) > 0,
            image_writer_processes=4,
            image_writer_threads=1,
        )

        # Copy each episode with the added 'next.reward' feature
        # Calculate total frames to process
        total_frames = sum(
            self.get_episode_frames(ep_idx)[1] - self.get_episode_frames(ep_idx)[0]
            for ep_idx in range(self.num_episodes)
        )

        print(f"\n  Copying {self.num_episodes} episodes ({total_frames} total frames)...")

        # Single progress bar for all frames
        pbar = tqdm(total=total_frames, desc="  Processing frames", unit="frame", leave=True)

        for ep_idx in range(self.num_episodes):
            start_idx, end_idx = self.get_episode_frames(ep_idx)

            # Get episode task
            first_frame = self.dataset[start_idx]
            task = first_frame["task"]

            # Add each frame with the reward
            for frame_idx in range(start_idx, end_idx):
                # Use dataset[idx] to get properly loaded frame (with images/videos)
                frame = self.dataset[frame_idx]

                # Convert to dict and remove metadata fields
                frame_dict = {}
                for key in self.dataset.features.keys():
                    if key not in [
                        "index",
                        "episode_index",
                        "frame_index",
                        "timestamp",
                        "task_index",
                        "task",
                    ]:
                        value = frame[key]
                        # Convert torch tensors to numpy
                        if isinstance(value, torch.Tensor):
                            value = value.cpu().numpy()

                        # Convert images from CHW to HWC format if needed
                        if key in self.dataset.meta.camera_keys and value.ndim == 3:
                            # Images come as (C, H, W) but LeRobot expects (H, W, C)
                            value = np.transpose(value, (1, 2, 0))

                        frame_dict[key] = value

                # Add reward (using "next.reward" for LeRobot visualization compatibility)
                frame_dict["next.reward"] = np.array([rewards[frame_idx]], dtype=np.float32)

                # Add frame to new dataset
                timestamp = frame["timestamp"].item()
                new_dataset.add_frame(frame_dict, task=task, timestamp=timestamp)

                # Update progress bar
                pbar.update(1)
                pbar.set_postfix({"episode": f"{ep_idx+1}/{self.num_episodes}"})

            # Save episode (this triggers video encoding which can be slow)
            pbar.set_description(f"  Saving episode {ep_idx+1}/{self.num_episodes}")
            new_dataset.save_episode()
            pbar.set_description("  Processing frames")

        pbar.close()

        print(f"\n✓ LeRobotDataset saved successfully!")
        print(f"  Location: {full_output_path}")
        print(f"  Total episodes: {new_dataset.num_episodes}")
        print(f"  Total frames: {new_dataset.num_frames}")

        # Save metadata
        metadata_path = full_output_path / "labeling_metadata.txt"
        with open(metadata_path, "w") as f:
            f.write(f"Original dataset: {self.dataset.repo_id}\n")
            f.write(f"Total frames: {len(self.dataset)}\n")
            f.write(f"Number of episodes: {self.num_episodes}\n")
            f.write(f"Reward range: [{rewards.min():.6f}, {rewards.max():.6f}]\n")
            f.write(f"\nSubtasks:\n")
            for i, name in enumerate(self.task_names):
                f.write(f"  {i}. {name}\n")
            f.write(f"\nBoundaries (marking END of subtasks):\n")
            for ep_idx in range(self.num_episodes):
                if self.boundaries.get(ep_idx):
                    f.write(f"  Episode {ep_idx}:\n")
                    for frame_idx, subtask_idx in self.boundaries[ep_idx]:
                        start_idx, _ = self.get_episode_frames(ep_idx)
                        frame_in_ep = frame_idx - start_idx
                        f.write(
                            f"    Frame {frame_in_ep} (global {frame_idx}): END of '{self.task_names[subtask_idx]}'\n"
                        )
                else:
                    f.write(f"  Episode {ep_idx}: No boundaries\n")
        print(f"    └── labeling_metadata.txt")

        print(f"\n💡 Load with:")
        print(f"   from lerobot.datasets.lerobot_dataset import LeRobotDataset")
        print(f'   ds = LeRobotDataset("{dataset_name}", root="{full_output_path}")')

        return full_output_path

    def run(self):
        """Main loop for interactive labeling."""
        # Initialize display
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 1280, 720)

        # Start at first frame of first episode
        start_idx, _ = self.get_episode_frames(self.current_episode)
        self.current_frame = start_idx

        print("\n" + "=" * 50)
        print("Dataset Labeler Started")
        print("=" * 50)
        print(f"\nSubtasks:")
        for i, name in enumerate(self.task_names):
            print(f"  {i+1}. {name}")
        print("\nPress keys to begin labeling...")

        try:
            while self.running:
                # Get current frame
                frame_data = self.dataset[self.current_frame]

                # Create display
                display_frame = self.create_display_frame(frame_data)
                cv2.imshow(self.window_name, display_frame)

                # Handle keyboard input from opencv
                key = cv2.waitKey(30) & 0xFF

                if key == ord("q"):
                    print("Quitting without saving...")
                    break
                elif key == ord("s"):
                    print("\nSaving and quitting...")
                    # Use fast save method that copies videos instead of re-encoding
                    self.save_dataset_with_rewards_fast()
                    break
                elif key == ord("n"):
                    self.next_episode()
                elif key == ord("p"):
                    self.prev_episode()
                elif key == ord("u"):
                    self.undo_last_boundary()
                elif ord("1") <= key <= ord("9"):
                    subtask_idx = key - ord("1")
                    self.mark_boundary(subtask_idx)
                elif key == 32:  # space
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Playing'}...")
                elif key == 81 or key == 83:  # Left arrow (81) or Right arrow (83) on some systems
                    if key == 81:  # Left arrow
                        self.move_frame(-1)
                    elif key == 83:  # Right arrow
                        self.move_frame(1)
                elif key == 27:  # ESC
                    print("ESC pressed, quitting without saving...")
                    break

                # Auto-advance if playing
                if not self.paused:
                    start_idx, end_idx = self.get_episode_frames(self.current_episode)
                    self.current_frame += self.playback_speed
                    if self.current_frame >= end_idx:
                        # Loop back or move to next episode
                        self.current_frame = start_idx

        finally:
            cv2.destroyAllWindows()
            print("\nLabeler closed.")


def main():
    """Main entry point."""

    labeler = DatasetLabeler(
        repo_id="ETHRC/piper_towel_v0",
        task_names=TASK_NAMES,
    )
    labeler.run()


if __name__ == "__main__":
    main()
