"""Common agent interface and hyper-parameter container."""

from __future__ import annotations

import abc
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from gymnasium import spaces

ALGORITHMS = ("sac", "td3", "ddpg", "ppo", "a2c")


@dataclass(frozen=True)
class AgentConfig:
    """Hyper-parameters of one agent.

    The first block is Table 1 of the paper (shared by all algorithms in the archived
    ``hyperparameter.json`` files); ``clip_grad_norm``, ``reward_scale`` and ``horizon_len``
    are additional keys of those files.  The remaining fields are not stated in the paper or
    the archive; their defaults are listed in ``docs/implementation_notes.md``.

    Attributes:
        algo: one of ``sac``, ``td3``, ``ddpg``, ``ppo``, ``a2c``.
        hidden_sizes: hidden layer widths (Table 1: 64 x 32).
        learning_rate: learning rate of every optimiser (Table 1: 1e-5).
        buffer_size: replay capacity (Table 1: 2^21); ignored by on-policy agents.
        batch_size: minibatch size (Table 1: 2^9; the archive uses 2^10 for A2C / PPO).
        gamma: discount factor (Table 1: 0.98).
        tau: soft-update rate of the target networks (Table 1: 0.005).
        clip_grad_norm: gradient-norm clip (archive: 3.0).
        reward_scale: factor applied to rewards before learning (archive: 1).
        horizon_len: environment steps collected between two update phases (archive: 256).
        repeat_times: gradient steps per collected step (off-policy) or reuse factor of a
            rollout (on-policy).
        learning_starts: off-policy only; uniformly random actions until this many steps.
        init_temperature: initial SAC temperature ``omega``.
        target_entropy: SAC entropy target; ``None`` means ``-dim(A)`` (below Eq. 24).
        explore_noise_std: Gaussian exploration noise of DDPG / TD3.
        policy_noise_std, policy_noise_clip, policy_delay: TD3 target smoothing and delay.
        gae_lambda, clip_ratio, entropy_coef: A2C / PPO settings.
    """

    algo: str = "sac"
    hidden_sizes: tuple[int, ...] = (64, 32)
    learning_rate: float = 1e-5
    buffer_size: int = 2**21
    batch_size: int = 2**9
    gamma: float = 0.98
    tau: float = 0.005
    clip_grad_norm: float = 3.0
    reward_scale: float = 1.0
    horizon_len: int = 256
    repeat_times: float = 1.0
    learning_starts: int = 1024
    init_temperature: float = math.exp(-1.0)
    target_entropy: float | None = None
    explore_noise_std: float = 0.1
    policy_noise_std: float = 0.2
    policy_noise_clip: float = 0.5
    policy_delay: int = 2
    gae_lambda: float = 0.95
    clip_ratio: float = 0.25
    entropy_coef: float = 0.01

    def __post_init__(self) -> None:
        object.__setattr__(self, "hidden_sizes", tuple(int(h) for h in self.hidden_sizes))
        if self.algo not in ALGORITHMS:
            raise ValueError(f"algo must be one of {ALGORITHMS}, got {self.algo!r}")
        if self.batch_size < 1 or self.horizon_len < 1 or self.buffer_size < 1:
            raise ValueError("batch_size, horizon_len and buffer_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["hidden_sizes"] = list(self.hidden_sizes)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown agent fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> AgentConfig:
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})


class Agent(abc.ABC):
    """Interface shared by all algorithms.

    The training loop calls :meth:`act` and :meth:`store` once per environment step and
    :meth:`update` after every ``horizon_len`` steps.  Actions live in ``[-1, 1]``.
    """

    on_policy: bool = False

    def __init__(
        self, obs_space: spaces.Box, action_space: spaces.Box, config: AgentConfig, seed: int = 0
    ) -> None:
        self.config = config
        self.obs_dim = int(np.prod(obs_space.shape))
        self.act_dim = int(np.prod(action_space.shape))
        self.obs_low = np.asarray(obs_space.low, dtype=np.float32)
        self.obs_high = np.asarray(obs_space.high, dtype=np.float32)
        self.seed = seed
        self.total_steps = 0

    @abc.abstractmethod
    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Action for one observation; ``deterministic=True`` is used for evaluation."""

    @abc.abstractmethod
    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        terminated: bool,
        truncated: bool,
    ) -> None:
        """Record the transition produced by the last call of :meth:`act`."""

    @abc.abstractmethod
    def update(self) -> dict[str, float]:
        """Run one update phase and return scalar diagnostics."""

    @abc.abstractmethod
    def modules(self) -> dict[str, torch.nn.Module]:
        """Named networks to checkpoint."""

    def state_dict(self) -> dict[str, Any]:
        """Checkpoint made only of tensors and plain numbers (loadable with weights_only)."""
        return {name: module.state_dict() for name, module in self.modules().items()}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        for name, module in self.modules().items():
            module.load_state_dict(state[name])

    def save(self, path: str | Path) -> None:
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path) -> None:
        self.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))

    def _clip(self, module_or_params: Any) -> None:
        params = (
            module_or_params.parameters()
            if isinstance(module_or_params, torch.nn.Module)
            else module_or_params
        )
        torch.nn.utils.clip_grad_norm_(params, self.config.clip_grad_norm)
