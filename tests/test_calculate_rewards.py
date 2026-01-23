import math

import pytest
import torch

from sarm.config.sarm_config import SarmConfig
from sarm.dataset.gap_dataset import GapLerobotDataset
from sarm.model.reward_sarm import RewardWeights, RewardSarm
import jax.numpy as jnp

from sarm.model.sarm import Sarm


def test_rewar_weights():
    rw = RewardWeights()
    assert rw.std == 0
    assert rw.mu == 0

    rewards_1 = jnp.array([1, 2])
    rw.update_stats(rewards=rewards_1)  # Capture the returned updated instance

    assert rw.mu == rewards_1.mean()
    assert rw.std == rewards_1.std(ddof=1)

    rewards_2 = jnp.array([3, 7])
    rw.update_stats(rewards=rewards_2)
    assert rw.mu == jnp.concatenate([rewards_1, rewards_2]).mean()
    assert math.isclose(rw.std, jnp.concatenate([rewards_1, rewards_2]).std(ddof=1), abs_tol=1e-6)


def test_mu_reward_clapping():
    rw = RewardWeights()
    rewards_1 = jnp.array([1, -2])
    rw.update_stats(rewards=rewards_1)
    assert rw.mu == 0.0


def test_get_weights():
    rw = RewardWeights()
    rewards = jnp.array([0.0, 0.0])
    weights = rw.get_weights(rewards)
    assert jnp.array_equal(jnp.array([0.0, 0.0]), weights)
    assert all(weights >= 0) and all(weights <= 1)

    rw.std = 0.05
    rw.mu = 0.1
    rw.kappa = float("inf")
    rw.epsilon = 0.0001
    rewards = jnp.array([0.1, 0.2])
    weights = rw.get_weights(rewards)
    expected_weights = (rewards - (0.1 - 2 * 0.05)) / (4 * 0.05 + rw.epsilon)
    assert jnp.array_equal(weights, expected_weights)
    assert all(weights >= 0) and all(weights <= 1)

    rw.std = 0.05
    rw.mu = 0.1
    rw.kappa = 0.15
    rw.epsilon = 0.0001
    rewards = jnp.array([0.1, 0.2])
    weights = rw.get_weights(rewards)
    expected_weights = jnp.array([(rewards[0] - (0.1 - 2 * 0.05)) / (4 * 0.05 + rw.epsilon), 1.0])
    assert jnp.array_equal(weights, expected_weights)
    assert all(weights >= 0) and all(weights <= 1)


class SarmMock:
    def __init__(self):
        self.n = 0

    def __call__(self, batch):
        self.n += 1
        shape = batch["observation.state"].shape
        return jnp.ones((shape[0], shape[1])) * self.n

@pytest.mark.parametrize('mock', [True, False])
def test_mocked_reward_model(mock):
    if mock:
        sarm_model = SarmMock()
    else:
        config = SarmConfig()
        sarm_model = Sarm.init_sarm_from_config(config=config, key=None)
    reward_model = RewardSarm(sarm=sarm_model)
    repo_id = "ETHRC/towel_base_with_rewards"
    dataset_gab = GapLerobotDataset(repo_id=repo_id, action_horizon=25, frame_gap=30, t_step_lookback=8)
    batch_size = 4
    data_loader = torch.utils.data.DataLoader(
        dataset_gab,
        batch_size=batch_size,
    )
    data_iter = iter(data_loader)
    batch = next(data_iter)
    weights = reward_model(batch)
    assert len(weights) == batch_size
    assert math.isclose(weights.sum(), 1, abs_tol=1e-5)

