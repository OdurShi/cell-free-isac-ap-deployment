"""Neural networks shared by the agents (hidden layers 64 x 32 by default, Table 1)."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def mlp(sizes: Sequence[int], output_activation: type[nn.Module] | None = None) -> nn.Sequential:
    """Fully connected network with ReLU between layers."""
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class ObsNormalizer(nn.Module):
    """Fixed affine map of the observation box onto ``[-1, 1]`` (no learned statistics)."""

    def __init__(self, low: np.ndarray, high: np.ndarray) -> None:
        super().__init__()
        low_t = torch.as_tensor(np.asarray(low), dtype=torch.float32)
        high_t = torch.as_tensor(np.asarray(high), dtype=torch.float32)
        self.register_buffer("center", 0.5 * (low_t + high_t))
        self.register_buffer("scale", 0.5 * (high_t - low_t))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return (obs - self.center) / self.scale


class SquashedGaussianActor(nn.Module):
    """SAC policy ``pi_mu(a | s)``: a diagonal Gaussian squashed by ``tanh``."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: Sequence[int], low: np.ndarray, high: np.ndarray
    ) -> None:
        super().__init__()
        self.normalizer = ObsNormalizer(low, high)
        self.trunk = mlp([obs_dim, *hidden], output_activation=nn.ReLU)
        self.mean_head = nn.Linear(hidden[-1], act_dim)
        self.log_std_head = nn.Linear(hidden[-1], act_dim)

    def distribution(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean and log standard deviation of the pre-squash Gaussian."""
        feat = self.trunk(self.normalizer(obs))
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return self.mean_head(feat), log_std

    def sample(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reparameterised action ``a ~ pi(. | s)`` and its log-probability.

        ``log pi(a | s) = log N(u; mean, std) - sum_i log(1 - tanh(u_i)^2)`` with ``a = tanh(u)``.
        """
        mean, log_std = self.distribution(obs)
        u = mean + log_std.exp() * torch.randn_like(mean)
        log_prob = _normal_log_prob(u, mean, log_std).sum(dim=-1)
        log_prob = log_prob - (2.0 * (math.log(2.0) - u - nn.functional.softplus(-2.0 * u))).sum(-1)
        return torch.tanh(u), log_prob

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action ``tanh(mean)`` used for evaluation."""
        return torch.tanh(self.distribution(obs)[0])


class DeterministicActor(nn.Module):
    """Deterministic policy of DDPG / TD3 with ``tanh`` output."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: Sequence[int], low: np.ndarray, high: np.ndarray
    ) -> None:
        super().__init__()
        self.normalizer = ObsNormalizer(low, high)
        self.net = mlp([obs_dim, *hidden, act_dim], output_activation=nn.Tanh)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(obs))


class GaussianActor(nn.Module):
    """A2C / PPO policy: Gaussian with state-independent log-std; ``tanh`` applied by the agent."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: Sequence[int], low: np.ndarray, high: np.ndarray
    ) -> None:
        super().__init__()
        self.normalizer = ObsNormalizer(low, high)
        self.net = mlp([obs_dim, *hidden, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Mean of the pre-squash Gaussian."""
        return self.net(self.normalizer(obs))

    def log_prob_entropy(
        self, obs: torch.Tensor, raw_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Log-probability of a pre-squash action and the entropy of the Gaussian."""
        mean = self.forward(obs)
        log_std = self.log_std.clamp(LOG_STD_MIN, LOG_STD_MAX).expand_as(mean)
        log_prob = _normal_log_prob(raw_action, mean, log_std).sum(dim=-1)
        entropy = (0.5 + 0.5 * math.log(2.0 * math.pi) + log_std).sum(dim=-1)
        return log_prob, entropy


class QNetwork(nn.Module):
    """State-action value ``Q_phi(s, a)``."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: Sequence[int], low: np.ndarray, high: np.ndarray
    ) -> None:
        super().__init__()
        self.normalizer = ObsNormalizer(low, high)
        self.net = mlp([obs_dim + act_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.normalizer(obs), action], dim=-1)).squeeze(-1)


class TwinQNetwork(nn.Module):
    """Two independent critics ``Q_phi1, Q_phi2`` (clipped double-Q)."""

    def __init__(
        self, obs_dim: int, act_dim: int, hidden: Sequence[int], low: np.ndarray, high: np.ndarray
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(obs_dim, act_dim, hidden, low, high)
        self.q2 = QNetwork(obs_dim, act_dim, hidden, low, high)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(obs, action), self.q2(obs, action)

    def min_q(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """The smaller of the two Q-values."""
        return torch.min(*self.forward(obs, action))


class ValueNetwork(nn.Module):
    """State value ``V(s)`` for A2C / PPO."""

    def __init__(self, obs_dim: int, hidden: Sequence[int], low: np.ndarray, high: np.ndarray):
        super().__init__()
        self.normalizer = ObsNormalizer(low, high)
        self.net = mlp([obs_dim, *hidden, 1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(self.normalizer(obs)).squeeze(-1)


def _normal_log_prob(x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    return -0.5 * ((x - mean) / log_std.exp()) ** 2 - log_std - 0.5 * math.log(2.0 * math.pi)


@torch.no_grad()
def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    """Soft target update ``phi_bar <- tau * phi + (1 - tau) * phi_bar`` (Section 3.2)."""
    for p_target, p_online in zip(target.parameters(), online.parameters(), strict=True):
        p_target.mul_(1.0 - tau).add_(p_online, alpha=tau)
