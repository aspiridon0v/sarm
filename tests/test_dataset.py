import pytest

from sarm.config.sarm_config import SarmConfig
from sarm.dataset.data_utils import get_valid_episodes, split_train_eval_episodes
from sarm.dataset.dataset import SarmDataset

config = SarmConfig()


@pytest.fixture(scope="module")
def config():
    return SarmConfig()


@pytest.fixture(scope="module")
def dataset(config: SarmConfig):

    valid_episodes = get_valid_episodes("ETHRC/piper_towel_v0_with_rewards")
    train_episodes, eval_episodes = split_train_eval_episodes(valid_episodes)

    train_dataset = SarmDataset(
        repo_id=config.general_config.repo_id_sparse,
        horizon=config.model_config.horizon,
        episodes=train_episodes,
        n_obs_steps=config.model_config.n_obs_steps,
        frame_gap=config.model_config.frame_gap,
        max_rewind_steps=config.model_config.max_rewind_steps,
        image_names=config.general_config.camera_names,
        annotation_list=config.model_config.sparse_annotation_list,
        task=config.general_config.task_name,
    )

    eval_dataset = SarmDataset(
        repo_id=config.general_config.repo_id_sparse,
        horizon=config.model_config.horizon,
        episodes=eval_episodes,
        n_obs_steps=config.model_config.n_obs_steps,
        frame_gap=config.model_config.frame_gap,
        max_rewind_steps=config.model_config.max_rewind_steps,
        image_names=config.general_config.camera_names,
        annotation_list=config.model_config.sparse_annotation_list,
        task=config.general_config.task_name,
    )

    return train_dataset, eval_dataset


def test_get_item(dataset: tuple[SarmDataset, SarmDataset], config: SarmConfig):
    train_dataset, eval_dataset = dataset

    # First item in the dataset,
    indices = [0, len(train_dataset) // 2, len(train_dataset) - 1]

    for idx in indices:

        train_item = train_dataset[idx]

        assert train_item["action"].shape == (
            config.model_config.horizon + 1,
            config.model_config.state_dim,
        )
        assert train_item["observation.state"].shape == (
            config.model_config.horizon + 1 + config.model_config.max_rewind_steps,
            config.model_config.state_dim,
        ), f"State shape is incorrect: {train_item['observation.state'].shape}"
        assert train_item["targets"].shape == (
            config.model_config.horizon + 1 + config.model_config.max_rewind_steps,
        ), f"Targets shape is incorrect: {train_item['targets'].shape}"
        assert train_item["frame_relative_indices"].shape == (
            config.model_config.horizon + 1 + config.model_config.max_rewind_steps,
        ), f"Frame relative indices shape is incorrect: {train_item['frame_relative_indices'].shape}"
        assert train_item["task"] == config.general_config.task_name
        for key in config.general_config.camera_names:
            assert train_item.get(key, None) is not None, f"Camera {key} is missing"
        assert (
            train_item["lengths"] >= config.model_config.horizon
            and train_item["lengths"]
            <= config.model_config.horizon + config.model_config.max_rewind_steps
        ), f"Lengths is incorrect: {train_item['lengths']}"
