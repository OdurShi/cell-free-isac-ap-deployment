"""Deterministic-policy baselines: DDPG and TD3."""

from __future__ import annotations

import copy

import numpy as np
import torch
from gymnasium import spaces

from cfisac.agents.base import Agent, AgentConfig
from cfisac.agents.buffer import ReplayBuffer
from cfisac.agents.networks import DeterministicActor, QNetwork, TwinQNetwork, soft_update


class TD3Agent(Agent):
    """Twin-delayed DDPG (Fujimoto et al., 2018).

    Clipped double-Q learning, target-policy smoothing and delayed actor / target updates.
    :class:`DDPGAgent` switches the three mechanisms off.
    """

    on_policy = False
    twin_critic = True
    smooth_target = True

    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Box,
        config: AgentConfig,
        seed: int = 0,
        buffer_capacity: int | None = None,
    ) -> None:
        super().__init__(obs_space, action_space, config, seed)
        args = (self.obs_dim, self.act_dim, config.hidden_sizes, self.obs_low, self.obs_high)
        self.actor = DeterministicActor(*args)
        self.actor_target = copy.deepcopy(self.actor).requires_grad_(False)
        self.critic = TwinQNetwork(*args) if self.twin_critic else QNetwork(*args)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        lr = config.learning_rate
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        capacity = config.buffer_size if buffer_capacity is None else buffer_capacity
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, capacity, seed)
        self._rng = np.random.default_rng(seed)
        self._new_steps = 0
        self._updates = 0

    @property
    def policy_delay(self) -> int:
        return max(1, self.config.policy_delay)

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        if not deterministic and self.total_steps < self.config.learning_starts:
            return self._rng.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
        action = self.actor(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))[0].numpy()
        if not deterministic:
            action = action + self._rng.normal(0.0, self.config.explore_noise_std, self.act_dim)
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        self.buffer.add(obs, action, reward * self.config.reward_scale, next_obs, terminated)
        self.total_steps += 1
        self._new_steps += 1

    def _q_min(self, critic: torch.nn.Module, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return critic.min_q(obs, act) if self.twin_critic else critic(obs, act)

    def update_step(self) -> dict[str, float]:
        """One critic step and, every ``policy_delay`` steps, one actor and target step."""
        cfg = self.config
        batch = self.buffer.sample(cfg.batch_size)
        with torch.no_grad():
            next_action = self.actor_target(batch["next_obs"])
            if self.smooth_target:
                noise = (torch.randn_like(next_action) * cfg.policy_noise_std).clamp(
                    -cfg.policy_noise_clip, cfg.policy_noise_clip
                )
                next_action = (next_action + noise).clamp(-1.0, 1.0)
            q_next = self._q_min(self.critic_target, batch["next_obs"], next_action)
            target = batch["reward"] + cfg.gamma * (1.0 - batch["terminal"]) * q_next
        if self.twin_critic:
            q1, q2 = self.critic(batch["obs"], batch["action"])
            critic_loss = ((q1 - target) ** 2).mean() + ((q2 - target) ** 2).mean()
        else:
            critic_loss = ((self.critic(batch["obs"], batch["action"]) - target) ** 2).mean()
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(self.critic)
        self.critic_opt.step()

        logs = {"critic_loss": critic_loss.item()}
        self._updates += 1
        if self._updates % self.policy_delay == 0:
            self.critic.requires_grad_(False)
            action = self.actor(batch["obs"])
            q_value = (
                self.critic.q1(batch["obs"], action)
                if self.twin_critic
                else self.critic(batch["obs"], action)
            )
            actor_loss = -q_value.mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip(self.actor)
            self.actor_opt.step()
            self.critic.requires_grad_(True)
            soft_update(self.actor_target, self.actor, cfg.tau)
            soft_update(self.critic_target, self.critic, cfg.tau)
            logs["actor_loss"] = actor_loss.item()
        return logs

    def update(self) -> dict[str, float]:
        new_steps, self._new_steps = self._new_steps, 0
        if len(self.buffer) < max(self.config.batch_size, self.config.learning_starts):
            return {}
        logs: dict[str, float] = {}
        for _ in range(max(1, int(new_steps * self.config.repeat_times))):
            logs.update(self.update_step())
        return logs

    def modules(self) -> dict[str, torch.nn.Module]:
        return {
            "actor": self.actor,
            "actor_target": self.actor_target,
            "critic": self.critic,
            "critic_target": self.critic_target,
        }


class DDPGAgent(TD3Agent):
    """Deep deterministic policy gradient (Lillicrap et al., 2016): single critic, no delay."""

    twin_critic = False
    smooth_target = False

    @property
    def policy_delay(self) -> int:
        return 1
