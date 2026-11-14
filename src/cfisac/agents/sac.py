"""Soft actor-critic for AP deployment (Section 3.2, Eqs. 21-24)."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from gymnasium import spaces

from cfisac.agents.base import Agent, AgentConfig
from cfisac.agents.buffer import ReplayBuffer
from cfisac.agents.networks import SquashedGaussianActor, TwinQNetwork, soft_update


class SACAgent(Agent):
    """SAC with twin critics, target critics and automatic temperature.

    Mapping to the paper:

    * maximum-entropy objective ``C(pi)``, Eq. (21) -> realised by the three losses below;
    * actor ``pi_mu`` and its loss ``L_pi(mu)``, Eq. (22) -> :meth:`_actor_loss`;
    * critics ``Q_phi1, Q_phi2`` with targets ``Q_phibar1, Q_phibar2`` and the soft Bellman
      residual ``L_Q(phi)``, Eq. (23) -> :meth:`_critic_loss`;
    * soft update ``phibar <- tau phi + (1 - tau) phibar`` -> :func:`soft_update`;
    * temperature ``omega`` and its loss ``L(omega)``, Eq. (24), with entropy target
      ``-dim(A)`` -> :meth:`_temperature_loss`.
    """

    on_policy = False

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
        args = (self.obs_dim, self.act_dim, hidden, self.obs_low, self.obs_high)
        self.actor = SquashedGaussianActor(*args)
        self.critic = TwinQNetwork(*args)
        self.critic_target = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_omega = torch.tensor(math.log(config.init_temperature), requires_grad=True)
        self.target_entropy = (
            -float(self.act_dim) if config.target_entropy is None else config.target_entropy
        )
        lr = config.learning_rate
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.omega_opt = torch.optim.Adam([self.log_omega], lr=lr)
        capacity = config.buffer_size if buffer_capacity is None else buffer_capacity
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, capacity, seed)
        self._rng = np.random.default_rng(seed)
        self._new_steps = 0

    @property
    def omega(self) -> torch.Tensor:
        """Temperature coefficient ``omega`` of Eq. (21)."""
        return self.log_omega.exp()

    # ----------------------------------------------------------- interaction
    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        if deterministic:
            return self.actor(obs_t)[0].numpy()
        if self.total_steps < self.config.learning_starts:
            return self._rng.uniform(-1.0, 1.0, size=self.act_dim).astype(np.float32)
        return self.actor.sample(obs_t)[0][0].numpy()

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        # A time-limit truncation is not a terminal state: the critic keeps bootstrapping.
        self.buffer.add(obs, action, reward * self.config.reward_scale, next_obs, terminated)
        self.total_steps += 1
        self._new_steps += 1

    # ---------------------------------------------------------------- losses
    def _critic_loss(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Soft Bellman residual, Eq. (23).

        ``y = r(s_i, a_i) + gamma * (min_j Q_phibar_j(s_{i+1}, a_{i+1})
        - omega * log pi_mu(a_{i+1} | s_{i+1}))`` with ``a_{i+1} ~ pi_mu(. | s_{i+1})`` and
        ``L_Q = E[1/2 (Q_phi_j(s_i, a_i) - y)^2]`` summed over both critics.
        """
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(batch["next_obs"])
            q_next = self.critic_target.min_q(batch["next_obs"], next_action)
            soft_value = q_next - self.omega * next_log_prob
            target = batch["reward"] + self.config.gamma * (1.0 - batch["terminal"]) * soft_value
        q1, q2 = self.critic(batch["obs"], batch["action"])
        return 0.5 * ((q1 - target) ** 2).mean() + 0.5 * ((q2 - target) ** 2).mean()

    def _actor_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy loss, Eq. (22): ``E[omega * log pi_mu(a | s) - Q_phi(s, a)]``, ``a ~ pi_mu``."""
        action, log_prob = self.actor.sample(batch["obs"])
        q_value = self.critic.min_q(batch["obs"], action)
        return (self.omega.detach() * log_prob - q_value).mean(), log_prob.detach()

    def _temperature_loss(self, log_prob: torch.Tensor) -> torch.Tensor:
        """Temperature loss, Eq. (24): ``E[-omega * log pi_mu(a | s) - omega * H_bar]``."""
        return (-self.omega * log_prob - self.omega * self.target_entropy).mean()

    # ---------------------------------------------------------------- update
    def update_step(self) -> dict[str, float]:
        """One gradient step on critics, actor and temperature, then the soft target update."""
        batch = self.buffer.sample(self.config.batch_size)

        critic_loss = self._critic_loss(batch)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self._clip(self.critic)
        self.critic_opt.step()

        self.critic.requires_grad_(False)  # the actor step differentiates Q w.r.t. the action only
        actor_loss, log_prob = self._actor_loss(batch)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self._clip(self.actor)
        self.actor_opt.step()
        self.critic.requires_grad_(True)

        omega_loss = self._temperature_loss(log_prob)
        self.omega_opt.zero_grad(set_to_none=True)
        omega_loss.backward()
        self.omega_opt.step()

        soft_update(self.critic_target, self.critic, self.config.tau)
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "omega": self.omega.item(),
            "entropy": -log_prob.mean().item(),
        }

    def update(self) -> dict[str, float]:
        new_steps, self._new_steps = self._new_steps, 0
        if len(self.buffer) < max(self.config.batch_size, self.config.learning_starts):
            return {}
        logs: dict[str, float] = {}
        for _ in range(max(1, int(new_steps * self.config.repeat_times))):
            logs = self.update_step()
        return logs

    def modules(self) -> dict[str, torch.nn.Module]:
        return {"actor": self.actor, "critic": self.critic, "critic_target": self.critic_target}

    def state_dict(self) -> dict:
        state = super().state_dict()
        state["log_omega"] = self.log_omega.detach().clone()
        return state

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        with torch.no_grad():
            self.log_omega.copy_(state["log_omega"])
