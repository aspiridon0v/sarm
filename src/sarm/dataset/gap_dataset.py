from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch


class GapLerobotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        action_horizon: int,
        t_step_lookback: int,
        frame_gap: int,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        action_keys=("action",),
    ):
        dataset_meta = LeRobotDatasetMetadata(repo_id, root)
        delta_timestamps = {key: [t / dataset_meta.fps for t in range(action_horizon)] for key in action_keys}
        self.action_horizon = action_horizon
        self.t_step_lookback = t_step_lookback
        self.frame_gap = frame_gap
        super().__init__(repo_id=repo_id, root=root, delta_timestamps=delta_timestamps, episodes=episodes)

    # Source: Taken from SARM paper supplementary materials
    def get_frame_indices(
        self,
        idx: int,
        n_obs_steps: int,
        frame_gap: int,
        ep_start: int = 0,
        ep_end: int | None = None,
    ) -> list[int]:
        """
        Build a monotonic sequence of length n_obs_steps+1 ending at idx.
        - Prefer fixed frame_gap when enough history exists.
        - Otherwise adapt the effective gap to fit within [ep_start, idx].
        - No padding; no extra inputs.

        Args:
            idx: last frame index (target frame).
            n_obs_steps: number of history steps (total length = n_obs_steps+1).
            frame_gap: desired fixed stride between history frames when possible.
            ep_start: episode start index (inclusive).
            ep_end: episode end index (inclusive); if None, unbounded above.

        Returns:
            List of indices (non-decreasing), length = n_obs_steps + 1.
        """
        # Clamp idx to episode bounds
        if ep_end is not None:
            idx = min(idx, ep_end)
        idx = max(idx, ep_start)

        gaps = n_obs_steps
        if gaps == 0:
            return [idx]

        # Check if fixed stride fits entirely inside the episode
        total_needed = frame_gap * gaps  # distance from earliest to idx
        available = idx - ep_start

        if available >= total_needed:
            # Use fixed frame_gap
            frames = [idx - frame_gap * (gaps - k) for k in range(gaps)] + [idx]
        else:
            # Not enough history: adapt stride by evenly spacing from ep_start to idx
            # Use integer rounding and enforce monotonicity.
            frames = [ep_start + round(available * k / gaps) for k in range(gaps)] + [idx]
            for i in range(1, len(frames)):
                if frames[i] < frames[i - 1]:
                    frames[i] = frames[i - 1]

        return frames

    def _get_hist_data(self, idx, current_ts, ep_idx):
        ep = self.meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        obs_indices = self.get_frame_indices(
            idx=idx,
            n_obs_steps=self.t_step_lookback,
            frame_gap=self.frame_gap,
            ep_start=ep_start,
            ep_end=ep_end,
        )

        # Convert absolute indices to relative indices for filtered dataset
        if self._absolute_to_relative_idx is not None:
            relative_obs_indices = [self._absolute_to_relative_idx[i] for i in obs_indices]
        else:
            relative_obs_indices = obs_indices

        item_hist = self.hf_dataset.select(relative_obs_indices)
        dict_hist = {k: torch.stack(list(item_hist[k])) for k in item_hist.features}

        task_index = torch.tensor(item_hist['task_index'])
        assert torch.all(task_index == task_index[0]), 'task should be the same for each index in the episode'
        task_str = self.meta.tasks.index[item_hist['task_index']].values[0]
        dict_hist['task_index'] = task_str  # keep for backward compat
        dict_hist['task'] = task_str        # alias for Sarm compatibility
        dict_hist['lengths'] = self.t_step_lookback + 1  # sequence length for Sarm

        if len(self.meta.video_keys) > 0:
            # query_indices uses absolute indices as _get_query_timestamps handles conversion
            query_indices = {key: obs_indices for key in self.meta.video_keys}
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            dict_hist = {**dict_hist, **video_frames}
        return dict_hist

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        ep_idx = int(item["episode_index"])
        idx_1 = int((item["action_is_pad"] == False).sum() + idx - 1)
        ts_0 = item["timestamp"].item()
        ts_1 = self.hf_dataset[idx]["timestamp"].item()

        dict_hist_0 = self._get_hist_data(idx=idx, current_ts=ts_0, ep_idx=ep_idx)
        dict_hist_1 = self._get_hist_data(idx=idx_1, current_ts=ts_1, ep_idx=ep_idx)


        for k in dict_hist_0:
            item[f"gap_data_0.{k}"] = dict_hist_0[k]
            item[f"gap_data_1.{k}"] = dict_hist_1[k]
        return item

