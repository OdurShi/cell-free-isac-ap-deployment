"""On-policy baselines: PPO and A2C."""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from cfisac.agents.base import Agent, AgentConfig
from cfisac.agents.buffer import RolloutBuffer
from cfisac.agents.networks import GaussianActor, ValueNetwork


class PPOAgent(Agent):
    """Proximal policy optimisation with the clipped surrogate (Schulman et al., 2017).

    A rollout of ``horizon_len`` steps is collected, advantages are estimated with GAE and
    ``max(1, horizon_len * repeat_times / batch_size)`` minibatch steps are taken; minibatches
    of ``batch_size`` are drawn with replacement, so that the archived setting
    ``batch_size = 1024 > horizon_len = 256`` is well defined.
    """

    on_policy = True
    clipped = True

    def __init__(
        self,
        obs_space: spaces.Box,
        action_space: spaces.Box,
        config: AgentConfig,
        seed: int = 0,
        buffer_capacity: int | None = None,
    ) -> None:
        super().__init__(obs_space, action_space, config, seed)
        hidden = config.hidden_sizes
        self.actor = GaussianActor(self.obs_dim, self.act_dim, hidden, self.obs_low, self.obs_high)
        self.critic = ValueNetwork(self.obs_dim, hidden, self.obs_low, self.obs_high)
        lr = config.learning_rate
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.buffer = RolloutBuffer(self.obs_dim, self.act_dim, config.horizon_len)
        self._generator = torch.Generator().manual_seed(seed)
        self._cache: tuple[np.ndarray, float, float] | None = None

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        mean = self.actor(obs_t)
        if deterministic:
            return torch.tanh(mean)[0].numpy()
        std = self.actor.log_std.exp()
        raw = mean + std * torch.randn(mean.shape, generator=self._generator)
        log_prob, _ = self.actor.log_prob_entropy(obs_t, raw)
        self._cache = (raw[0].numpy(), float(log_prob), float(self.critic(obs_t)))
        return torch.tanh(raw)[0].numpy()

    @torch.no_grad()
    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if self._cache is None:
            raise RuntimeError("store() must follow a stochastic act() call")
        raw, log_prob, value = self._cache
        self._cache = None
        next_value = float(self.critic(torch.as_tensor(next_obs, dtype=torch.float32)[None]))
        self.buffer.add(
            obs, raw, reward * self.config.reward_scale, value, log_prob,
            terminated, truncated, next_value,
        )
        self.total_steps += 1

    def _policy_loss(
        self, log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor
    ) -> torch.Tensor:
        ratio = (log_prob - old_log_prob).exp()
        clip = self.config.clip_ratio
        surrogate = torch.min(ratio * advantage, ratio.clamp(1.0 - clip, 1.0 + clip) * advantage)
        return -surrogate.mean()

    def update(self) -> dict[str, float]:
        if not self.buffer.full:
            return {}
        cfg = self.config
        self.buffer.finish(cfg.gamma, cfg.gae_lambda)
        data = self.buffer.tensors()
        n = len(self.buffer)
        advantage = data["advantage"]
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        logs: dict[str, float] = {}
        for _ in range(max(1, int(n * cfg.repeat_times / cfg.batch_size))):
            idx = torch.randint(n, (cfg.batch_size,), generator=self._generator)
            obs = data["obs"][idx]
            log_prob, entropy = self.actor.log_prob_entropy(obs, data["raw_action"][idx])
            actor_loss = self._policy_loss(log_prob, data["log_prob"][idx], advantage[idx])
            actor_loss = actor_loss - cfg.entropy_coef * entropy.mean()
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            self._clip(self.actor)
            self.actor_opt.step()

            critic_loss = ((self.critic(obs) - data["returns"][idx]) ** 2).mean()
            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            self._clip(self.critic)
            self.critic_opt.step()
            logs = {
                "actor_loss": actor_loss.item(),
                "critic_loss": critic_loss.item(),
                "entropy": entropy.mean().item(),
            }
        self.buffer.reset()
        return logs

    def modules(self) -> dict[str, torch.nn.Module]:
        return {"actor": self.actor, "critic": self.critic}


class A2CAgent(PPOAgent):
    """Advantage actor-critic: the plain policy-gradient loss ``-E[A * log pi]`` on each rollout."""

    clipped = False

    def _policy_loss(
        self, log_prob: torch.Tensor, old_log_prob: torch.Tensor, advantage: torch.Tensor
    ) -> torch.Tensor:
        return -(advantage * log_prob).mean()
