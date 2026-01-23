import logging
import traceback
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import torch

from sarm.dataset.gap_dataset import GapLerobotDataset

matplotlib.use("Agg")  # Non-interactive backend
import cv2
from tqdm import tqdm

from sarm.config.sarm_config import SarmConfig
from sarm.dataset.data_utils import get_valid_episodes
from sarm.dataset.dataset import SarmDataset
from sarm.model.sarm import Sarm
from sarm.utils.logger_setup import setup_logger
import equinox as eqx
from typing import Dict, Any, List



def adapt_gap_batch_sarm(gap_data: Dict[str, Any], prefix: str = "gap_data_0.") -> Dict[str, Any]:
    """Convert GAP dataset entry into a batched Sarm input dict."""
    sarm_data = {
        k.replace(prefix, ""): v.unsqueeze(0) if isinstance(v, torch.Tensor) else v
        for k, v in gap_data.items()
        if prefix in k
    }

    # Ensure batch-shaped metadata for Sarm.__call__.
    if isinstance(sarm_data.get("task"), str):
        sarm_data["task"] = [sarm_data["task"]]

    if not isinstance(sarm_data.get("lengths"), torch.Tensor):
        sarm_data["lengths"] = torch.tensor([sarm_data["lengths"]], dtype=torch.long)

    return sarm_data


def create_progress_plot(predictions, ground_truths, current_idx, figsize=(10, 3)):
    """Create a progress plot showing predictions vs ground truth."""
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(ground_truths))
    ax.plot(x, ground_truths, "g-", label="Ground Truth", linewidth=2)
    ax.plot(x[: current_idx + 1], predictions[: current_idx + 1], "b-", label="Prediction", linewidth=2)

    # Add current position marker
    if current_idx < len(predictions):
        ax.axvline(x=current_idx, color="r", linestyle="--", alpha=0.5)

    ax.set_xlabel("Frame Index")
    ax.set_ylabel("Progress")
    ax.set_ylim(-0.1, 1.1)  # Assuming progress is in range [0, 1]
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_title("Progress: Prediction vs Ground Truth")

    # Convert plot to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    img = img[:, :, :3]  # Remove alpha channel
    plt.close(fig)

    return img


def create_summary_plot(predictions, ground_truths, episode_idx, save_path, logger):
    """Create and save a summary plot for the entire episode."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    x = np.arange(len(ground_truths))

    # Plot 1: Predictions and Ground Truth over time
    ax1.plot(x, ground_truths, "g-", label="Ground Truth", linewidth=2)
    ax1.plot(x, predictions, "b-", label="Prediction", linewidth=2)
    ax1.set_xlabel("Frame Index")
    ax1.set_ylabel("Progress")
    ax1.set_title(f"Episode {episode_idx}: Progress Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Scatter plot of Predicted vs Ground Truth
    ax2.scatter(ground_truths, predictions, alpha=0.6)
    min_val = min(ground_truths.min(), predictions.min())
    max_val = max(ground_truths.max(), predictions.max())
    ax2.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")
    ax2.set_xlabel("Ground Truth Progress")
    ax2.set_ylabel("Predicted Progress")
    ax2.set_title(f"Episode {episode_idx}: Predicted vs Ground Truth")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Calculate and show MAE
    mae = np.abs(predictions - ground_truths).mean()
    ax2.text(
        0.05,
        0.95,
        f"MAE: {mae:.4f}",
        transform=ax2.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved summary plot to {save_path}")


def create_video(frames, predictions, ground_truths, episode_idx, save_path, fps=10, logger=None):
    """Create a video combining topdown images with progress plots."""
    if len(frames) == 0:
        if logger:
            logger.warning(f"No frames to create video for episode {episode_idx}")
        return

    # Get dimensions
    frame_h, frame_w = frames[0].shape[:2]

    # Create a sample plot to get dimensions
    sample_plot = create_progress_plot(predictions, ground_truths, 0)
    plot_h, plot_w = sample_plot.shape[:2]

    # Calculate output video dimensions
    # Scale plot to match frame width
    scale = frame_w / plot_w
    new_plot_h = int(plot_h * scale)
    new_plot_w = frame_w

    output_h = frame_h + new_plot_h
    output_w = frame_w

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(str(save_path), fourcc, fps, (output_w, output_h))

    if logger:
        logger.info(f"Creating video for episode {episode_idx} with {len(frames)} frames...")

    for i in tqdm(range(len(frames)), desc=f"Episode {episode_idx}"):
        # Get topdown frame (convert from RGB to BGR for opencv)
        frame_rgb = frames[i]

        # Create progress plot
        plot_img = create_progress_plot(predictions, ground_truths, i)

        # Resize plot to match frame width
        plot_resized = cv2.resize(plot_img, (new_plot_w, new_plot_h))

        # Combine frame and plot vertically
        combined = np.vstack([frame_rgb, plot_resized])

        # Convert RGB to BGR for opencv
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

        video_writer.write(combined_bgr)

    video_writer.release()
    if logger:
        logger.info(f"Saved video to {save_path}")


def process_single_frame(
    data_point,
    gap_data,
    config,
    sarm:Sarm,
):
    sarm_data_0 = adapt_gap_batch_sarm(gap_data, prefix="gap_data_0.")

    final_pred = sarm(sarm_data_0).squeeze(0)[-1]
    # Get ground truth
    progress = jnp.array(data_point["targets"].unsqueeze(0))

    gt_progress = float(progress[0, config.model_config.n_obs_steps])

    #normalize
    gt_progress = sarm.normalize_rewards(gt_progress)
    final_pred = sarm.normalize_rewards(final_pred)

    # Get topdown image at observation step
    topdown_idx = config.general_config.camera_names.index("observation.images.topdown")
    topdown_img = data_point[config.general_config.camera_names[topdown_idx]][config.model_config.n_obs_steps]
    topdown_img = topdown_img.numpy().transpose(1, 2, 0)  # C, H, W -> H, W, C
    topdown_img = (topdown_img * 255).astype(np.uint8)  # Assuming normalized to [0, 1]

    return final_pred, gt_progress, topdown_img




def eval(config: SarmConfig, num_episodes: int = 5, eval_frame_gap: int = 5):
    """Evaluate the model and generate videos."""
    logger = logging.getLogger(__name__)

    logger.info("Loading models...")
    sarm = Sarm.load_sarm_checkpoint_from_config(config)
    sarm = eqx.nn.inference_mode(sarm)

    logger.info("Loading datasets...")
    valid_episodes = get_valid_episodes(config.general_config.repo_id_sparse)
    dataset_val = SarmDataset(
        repo_id=config.general_config.repo_id_sparse,
        horizon=config.model_config.horizon,
        episodes=valid_episodes,
        n_obs_steps=config.model_config.n_obs_steps,
        frame_gap=config.model_config.frame_gap,
        max_rewind_steps=config.model_config.max_rewind_steps,
        image_names=config.general_config.camera_names,
        annotation_list=config.model_config.sparse_annotation_list,
        task=config.general_config.task_name,
        video_eval=True,
    )

    dataset_gap = GapLerobotDataset(repo_id=config.general_config.repo_id_sparse,
                                    action_horizon=config.model_config.horizon,
                                    t_step_lookback=config.model_config.n_obs_steps,
                                    frame_gap=config.model_config.frame_gap,
                                    episodes=valid_episodes)




    # Create output directory
    output_dir = Path("evaluation_results")
    output_dir.mkdir(exist_ok=True)
    logger.info(f"Saving results to {output_dir}")

    # Process episodes
    num_episodes = min(num_episodes, len(dataset_val.meta.episodes))
    logger.info(f"Processing {num_episodes} episodes...")

    for ep_idx in range(num_episodes):
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing Episode {ep_idx}")
        logger.info(f"{'='*50}")

        ep = dataset_val.meta.episodes[ep_idx]
        start_idx = ep["dataset_from_index"]
        end_idx = ep["dataset_to_index"]

        predictions = []
        ground_truths = []
        topdown_frames = []

        # Process all frames in the episode
        for idx in tqdm(range(start_idx, end_idx, eval_frame_gap), desc=f"Episode {ep_idx}"):
            try:
                data_point = dataset_val[idx]
                gap_data = dataset_gap[idx]
                pred, gt, topdown_img = process_single_frame(
                    data_point,
                    gap_data,
                    config,
                    sarm
                )
                predictions.append(pred)
                ground_truths.append(gt)
                topdown_frames.append(topdown_img)

            except Exception as e:
                logger.warning(f"Error processing frame {idx}: {e}")
                traceback.print_exception(e)
                continue

        if len(predictions) == 0:
            logger.warning(f"No valid predictions for episode {ep_idx}, skipping...")
            continue

        # Convert to numpy arrays
        predictions = np.array(predictions)
        ground_truths = np.array(ground_truths)

        # Calculate MAE for this episode
        mae = np.abs(predictions - ground_truths).mean()
        logger.info(f"Episode {ep_idx} MAE: {mae:.4f}")

        # Create video
        video_path = output_dir / f"episode_{ep_idx}_video.mp4"
        create_video(topdown_frames, predictions, ground_truths, ep_idx, video_path, fps=10, logger=logger)

        # Create summary plot
        plot_path = output_dir / f"episode_{ep_idx}_summary.png"
        create_summary_plot(predictions, ground_truths, ep_idx, plot_path, logger)

    logger.info(f"\n{'='*50}")
    logger.info("Evaluation complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info(f"{'='*50}")


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    config = SarmConfig()
    setup_logger(config, logger)

    # Evaluate on all episodes with frame gap of 5
    # num_episodes will be capped to total valid episodes in dataset
    eval(config, num_episodes=999, eval_frame_gap=5)
