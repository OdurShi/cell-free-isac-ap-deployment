"""Experience storage: replay buffer (off-policy) and rollout buffer with GAE (on-policy)."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Uniform FIFO replay buffer ``B`` of transitions ``(s_i, a_i, r_i, s_{i+1}, terminal)``."""

    def __init__(self, obs_dim: int, act_dim: int, capacity: int, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((self.capacity, act_dim), dtype=np.float32)
        self.reward = np.zeros(self.capacity, dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.terminal = np.zeros(self.capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self._size

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
    ) -> None:
        """Store one transition, overwriting the oldest one when full."""
        i = self._pos
        self.obs[i] = obs
        self.action[i] = action
        self.reward[i] = reward
        self.next_obs[i] = next_obs
        self.terminal[i] = float(terminated)
        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Draw a batch uniformly at random (with replacement)."""
        if self._size == 0:
            raise ValueError("cannot sample from an empty buffer")
        idx = self._rng.integers(0, self._size, size=batch_size)
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "action": torch.from_numpy(self.action[idx]),
            "reward": torch.from_numpy(self.reward[idx]),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "terminal": torch.from_numpy(self.terminal[idx]),
        }


class RolloutBuffer:
    """Fixed-length on-policy rollout with generalised advantage estimation."""

    def __init__(self, obs_dim: int, act_dim: int, length: int) -> None:
        self.length = int(length)
        self.obs = np.zeros((self.length, obs_dim), dtype=np.float32)
        self.raw_action = np.zeros((self.length, act_dim), dtype=np.float32)
        self.reward = np.zeros(self.length, dtype=np.float32)
        self.value = np.zeros(self.length, dtype=np.float32)
        self.log_prob = np.zeros(self.length, dtype=np.float32)
        self.terminated = np.zeros(self.length, dtype=bool)
        self.episode_end = np.zeros(self.length, dtype=bool)
        self.bootstrap = np.zeros(self.length, dtype=np.float32)
        self.advantage = np.zeros(self.length, dtype=np.float32)
        self.returns = np.zeros(self.length, dtype=np.float32)
        self._pos = 0

    def __len__(self) -> int:
        return self._pos

    @property
    def full(self) -> bool:
        return self._pos >= self.length

    def reset(self) -> None:
        self._pos = 0

    def add(
        self,
        obs: np.ndarray,
        raw_action: np.ndarray,
        reward: float,
        value: float,
        log_prob: float,
        terminated: bool,
        truncated: bool,
        next_value: float,
    ) -> None:
        """Store one step.

        ``next_value`` is ``V(s_{i+1})`` of the state actually reached; it is used to
        bootstrap when the episode was cut by the time limit and at the end of the rollout.
        """
        i = self._pos
        self.obs[i] = obs
        self.raw_action[i] = raw_action
        self.reward[i] = reward
        self.value[i] = value
        self.log_prob[i] = log_prob
        self.terminated[i] = terminated
        self.episode_end[i] = terminated or truncated
        self.bootstrap[i] = next_value
        self._pos += 1

    def finish(self, gamma: float, gae_lambda: float) -> None:
        """Compute GAE advantages and lambda-returns for the stored steps."""
        n = self._pos
        adv = 0.0
        for t in reversed(range(n)):
            if self.terminated[t]:
                next_value, next_adv = 0.0, 0.0
            elif self.episode_end[t] or t == n - 1:
                next_value, next_adv = float(self.bootstrap[t]), 0.0
            else:
                next_value, next_adv = float(self.value[t + 1]), adv
            delta = self.reward[t] + gamma * next_value - self.value[t]
            adv = delta + gamma * gae_lambda * next_adv
            self.advantage[t] = adv
        self.returns[:n] = self.advantage[:n] + self.value[:n]

    def tensors(self) -> dict[str, torch.Tensor]:
        """All stored steps as tensors."""
        n = self._pos
        return {
            "obs": torch.from_numpy(self.obs[:n].copy()),
            "raw_action": torch.from_numpy(self.raw_action[:n].copy()),
            "log_prob": torch.from_numpy(self.log_prob[:n].copy()),
            "advantage": torch.from_numpy(self.advantage[:n].copy()),
            "returns": torch.from_numpy(self.returns[:n].copy()),
        }
