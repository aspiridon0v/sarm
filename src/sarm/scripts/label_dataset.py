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

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


class BoundaryPredictor:
    def __init__(self, ep_starts, q=0.2, min_samples=0):
        self.ep_starts = ep_starts
        self.durations = defaultdict(list)
        self.q = q
        self.min_samples = min_samples

    def update(self, boundaries):
        self.durations = defaultdict(list)
        for ep_idx, bounds in boundaries.items():
            if len(bounds) < 2:
                continue
            for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
                if i == 0:
                    self.durations[i].append(a[0] - self.ep_starts[ep_idx])
                self.durations[i + 1].append(b[0] - a[0])

    def quantile_duration(self, durations):
        if durations == [] or len(durations) < self.min_samples:
            return 0
        return int(np.quantile(durations, self.q))

    def predict(self, current_bounds):
        if current_bounds == []:
            return self.quantile_duration(self.durations[0])
        sorted_bounds = sorted(current_bounds, key=lambda x: x[0])
        last_frame, last_id = sorted_bounds[-1]
        next_id = last_id + 1
        if self.durations[next_id] == []:
            return 0
        else:
            return self.quantile_duration(self.durations[next_id])

    def predict_subtask_end(self, current_bounds, ep_start):
        last_idx = np.max([b[0] for b in current_bounds] + [ep_start])
        sub_taks_lengh = self.predict(current_bounds=current_bounds)
        return int(sub_taks_lengh + last_idx)


class DatasetLabeler:
    """Interactive labeler for robot datasets with subtask boundary annotation."""

    def __init__(
        self,
        repo_id: str,
        task_names: List[str],
        camera_keys: List[str],
        boundaries: Optional[Dict[int, List[tuple]]] = None,
    ):
        self.dataset = LeRobotDataset(repo_id=repo_id)
        self.task_names = task_names
        self.camera_keys = camera_keys
        self.num_subtasks = len(task_names)

        # Get episode information from lerobot dataset
        self.num_episodes = self.dataset.num_episodes

        print(f"Dataset loaded: {repo_id}")
        print(f"Number of episodes: {self.num_episodes}")
        print(f"Total frames: {len(self.dataset)}")

        # Labeling state
        self.current_episode = 0
        self.current_frame = 0
        self.paused = True
        self.playback_speed = 1  # frames per update

        # Subtask boundaries: dict[episode_idx] -> list of (frame_idx, subtask_idx)
        self.boundaries = {ep: [] for ep in range(self.num_episodes)} if boundaries is None else boundaries

        # Boundary predictor
        self.predictor = BoundaryPredictor([self.get_episode_frames(ep)[0] for ep in range(self.num_episodes)])
        self.fast_mode = True
        # Running state
        self.running = True

        # Window name
        self.window_name = "Dataset Labeler"

    def get_episode_frames(self, episode_idx: int):
        """Get start and end frame indices for an episode."""

        ep = self.dataset.meta.episodes[episode_idx]
        start_idx = ep["dataset_from_index"]
        end_idx = ep["dataset_to_index"]

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
        for cam_key in self.camera_keys:
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
        info_panel_height = 200
        info_panel = np.zeros((info_panel_height, camera_view.shape[1], 3), dtype=np.uint8)

        # Add episode and frame info to top panel
        start_idx, end_idx = self.get_episode_frames(self.current_episode)
        frame_in_episode = self.current_frame - start_idx
        total_frames_in_episode = end_idx - start_idx

        episode_info = [
            f"Episode: {self.current_episode}/{self.num_episodes-1}  |"
            f"  Frame: {frame_in_episode}/{total_frames_in_episode-1} (Global: {self.current_frame})  |"
            f"  Status: {'PAUSED' if self.paused else 'PLAYING'}  |"
            f"  Speed: {self.playback_speed}x {'(Fast Mode)' if self.fast_mode else ''}"
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
            cv2.putText(info_panel, boundary_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            for i, (frame_idx, subtask_idx) in enumerate(self.boundaries[self.current_episode]):
                start_idx, _ = self.get_episode_frames(self.current_episode)
                frame_in_ep = frame_idx - start_idx
                boundary_info = f"  {i+1}. Frame {frame_in_ep}: END '{self.task_names[subtask_idx]}'"
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
        controls_panel_height = 160
        controls_panel = np.zeros((controls_panel_height, camera_view.shape[1], 3), dtype=np.uint8)

        controls_text = [
            "Controls:",
            "  SPACE: Play/Pause  |  UP: Jump to predicted boundary  |  F: Toggle fast mode",
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
                (10, 20 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
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
        if 0 <= subtask_idx < self.num_subtasks:
            self.boundaries[self.current_episode].append((self.current_frame, subtask_idx))
            self.boundaries[self.current_episode].sort(key=lambda x: x[0])
            self.predictor.update(self.boundaries)
            print(
                f"Marked END of '{self.task_names[subtask_idx]}' at frame {self.current_frame} "
                f"(Episode {self.current_episode})"
            )
            if subtask_idx + 1 < self.num_subtasks:
                print(f"  → Next subtask '{self.task_names[subtask_idx + 1]}' starts at frame {self.current_frame + 1}")

    def undo_last_boundary(self):
        if self.current_episode in self.boundaries and self.boundaries[self.current_episode]:
            removed = self.boundaries[self.current_episode].pop()
            self.predictor.update(self.boundaries)
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

    def save_dataset_with_rewards(self, output_dir: str = "data"):
        """Save the dataset with 'next.reward' using lerobot's modify_features utility.

        This is much faster than save_dataset_with_rewards() since it avoids re-encoding videos.

        Args:
            output_dir: Directory to save the new dataset (default: "data")
        """
        from pathlib import Path
        from lerobot.datasets.dataset_tools import modify_features

        # Compute rewards
        print("\n  Computing rewards...")
        rewards = self.compute_rewards()
        print(f"  ✓ Rewards computed for {len(rewards)} frames")

        print("\n" + "=" * 50)
        print("Fast Save: Using modify_features to add rewards...")
        print("=" * 50)

        # Dataset name
        dataset_name = self.dataset.repo_id.split("/")[-1] + "_with_rewards"
        output_path = Path(output_dir) / dataset_name

        print(f"\nCreating new LeRobotDataset: {dataset_name}")
        print(f"  Output: {output_path}")
        print(f"  Adding 'next.reward' feature")
        print(f"  Reward range: [{rewards.min():.3f}, {rewards.max():.3f}]")

        # Use modify_features to add the reward feature
        # Create a callable that returns the reward for each frame
        # The callable signature is: (row_dict, episode_idx, frame_in_episode) -> value
        def get_reward(row_dict, episode_idx, frame_in_episode):
            # Get the global frame index from the row
            global_idx = row_dict["index"]
            return float(rewards[global_idx])

        new_dataset = modify_features(
            dataset=self.dataset,
            add_features={"next.reward": (get_reward, {"dtype": "float32", "shape": (1,), "names": None})},
            output_dir=output_path,
            repo_id=dataset_name,
        )

        print(f"\n✓ Dataset saved successfully (no video re-encoding needed)!")
        print(f"  Location: {new_dataset.root}")

        # Save metadata
        metadata_path = new_dataset.root / "labeling_metadata.txt"
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
        print(f'   ds = LeRobotDataset("{dataset_name}", root="{new_dataset.root}")')

        return new_dataset.root

    @staticmethod
    def load_boundaries_from_metadata(metadata_path: str) -> Dict[int, List[tuple]]:
        """Load boundaries from a labeling_metadata.txt file.

        Args:
            metadata_path: Path to the labeling_metadata.txt file

        Returns:
            Dictionary mapping episode_idx -> list of (frame_idx, subtask_idx) tuples
        """
        import re

        boundaries = {}
        task_names_map = {}  # Maps task name string -> subtask_idx
        current_episode = None

        with open(metadata_path, "r") as f:
            lines = f.readlines()

        # First pass: Parse the Subtasks section to build the task name mapping
        in_subtasks_section = False
        in_boundaries_section = False

        for line in lines:
            # Parse subtasks section: "  0. Grasp right corner"
            if "Subtasks:" in line:
                in_subtasks_section = True
                in_boundaries_section = False
                continue

            if in_subtasks_section and not in_boundaries_section:
                subtask_match = re.match(r"\s*(\d+)\.\s+(.+)", line)
                if subtask_match:
                    subtask_idx = int(subtask_match.group(1))
                    task_name = subtask_match.group(2).strip()
                    task_names_map[task_name] = subtask_idx
                    continue

            # Find the "Boundaries" section
            if "Boundaries (marking END of subtasks):" in line:
                in_boundaries_section = True
                in_subtasks_section = False
                continue

            if not in_boundaries_section:
                continue

            # Parse episode line: "  Episode 0:"
            episode_match = re.match(r"\s*Episode (\d+):", line)
            if episode_match:
                current_episode = int(episode_match.group(1))
                boundaries[current_episode] = []
                continue

            # Parse "No boundaries" line
            if "No boundaries" in line:
                continue

            # Parse boundary line: "    Frame 45 (global 123): END of 'Grasp right corner'"
            boundary_match = re.match(r"\s*Frame \d+ \(global (\d+)\): END of \'(.+)\'", line)
            if boundary_match and current_episode is not None:
                global_frame_idx = int(boundary_match.group(1))
                task_name = boundary_match.group(2)

                # Look up the subtask index from the task name
                if task_name in task_names_map:
                    subtask_idx = task_names_map[task_name]
                    boundaries[current_episode].append((global_frame_idx, subtask_idx))
                else:
                    print(f"Warning: Unknown task name '{task_name}' in boundaries section")

        return boundaries

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
                    self.save_dataset_with_rewards()
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
                elif key == ord("f"):  # F key
                    self.fast_mode = not self.fast_mode
                    print(f"Fast mode: {'ON' if self.fast_mode else 'OFF'}")
                elif key == 82:  # Up arrow
                    _, ep_end = self.get_episode_frames(self.current_episode)
                    predicted = self.predictor.predict(self.boundaries[self.current_episode])
                    if predicted is not None:
                        self.move_frame(predicted)
                elif key == 81 or key == 83:  # Left arrow (81) or Right arrow (83)
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
                    self.playback_speed = 1
                    if self.fast_mode:
                        predicted = self.predictor.predict_subtask_end(self.boundaries[self.current_episode], start_idx)
                        if predicted - self.current_frame > 15:
                            self.playback_speed = 3

                    self.current_frame += self.playback_speed
                    if self.current_frame >= end_idx:
                        # Loop back or move to next episode
                        self.current_frame = start_idx

        finally:
            cv2.destroyAllWindows()
            print("\nLabeler closed.")


def main():
    """Main entry point."""

    TASK_NAMES = [
        "Grasp right corner",
        "Grasp left corner",
        "Fold towel horizontally",
        "Grasp right edge",
        "Fold towel vertically",
    ]

    # Camera keys in the dataset
    CAMERA_KEYS = [
        "observation.images.left_wrist",
        "observation.images.right_wrist",
        "observation.images.topdown",
    ]

    boundaries = None
    # Uncomment to load existing labels
    # boundaries = DatasetLabeler.load_boundaries_from_metadata('data/labeling_metadata.txt')

    labeler = DatasetLabeler(
        repo_id="ETHRC/towel_base",
        task_names=TASK_NAMES,
        camera_keys=CAMERA_KEYS,
        boundaries=boundaries,
    )
    labeler.run()


if __name__ == "__main__":
    main()
