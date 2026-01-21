import jax
import jax.numpy as jnp


class RewardWeights:
    def __init__(self, epsilon=1e-6, kappa=0.01):

        self.mu = jnp.array(0, dtype=jnp.float32)
        self.std = jnp.array(0, dtype=jnp.float32)
        self.M2 = jnp.array(0, dtype=jnp.float32)
        self.N = jnp.array(0, dtype=jnp.uint64)

        self.epsilon = epsilon
        self.kappa = kappa

    def update_stats(self, rewards: jax.Array):
        assert rewards.ndim == 1
        n = rewards.shape[0]
        N_new = self.N + n

        mu_b = rewards.mean()
        var_b = rewards.var(ddof=0)
        d_mu = mu_b - self.mu

        self.mu = self.mu + n * d_mu / N_new
        self.M2 = self.M2 + n * var_b + self.N * n * d_mu**2 / N_new
        self.std = jnp.sqrt(self.M2 / jnp.maximum(N_new - 1, 1))
        self.N = N_new

        # clamp mu
        self.mu = jnp.maximum(0.0, self.mu)

    def get_weights(self, rewards: jax.Array):
        norm_rewards = (rewards - (self.mu - 2 * self.std)) / (4 * self.std + self.epsilon)
        w = jnp.clip(norm_rewards, 0, 1)
        w_prior = jnp.where(rewards > self.kappa, 1, w)
        return w_prior


class RewardSarm:
    def __init__(self, sarm, epsilon=1e-6, kappa=0.01):
        self.sarm = sarm
        self.epsilon = epsilon
        self.rw = RewardWeights(epsilon=epsilon, kappa=kappa)

    def __call__(self, batch):
        sarm_data_0 = {k.replace("gap_data_0.", ""): v for k, v in batch.items() if "gap_data_0." in k}
        rewards_0 = self.sarm(sarm_data_0)  # B, T

        sarm_data_1 = {k.replace("gap_data_1.", ""): v for k, v in batch.items() if "gap_data_1." in k}
        rewards_1 = self.sarm(sarm_data_1)  # B, T

        r_hat = rewards_1[:, -1] - rewards_0[:, -1]  # B

        self.rw.update_stats(r_hat)
        weights = self.rw.get_weights(r_hat)
        weight_loss = weights / (weights.sum() + self.epsilon)

        return jax.lax.stop_gradient(weight_loss)
