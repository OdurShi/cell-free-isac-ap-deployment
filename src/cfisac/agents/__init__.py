"""Reinforcement-learning agents: SAC (the paper's method) and the baselines of Fig. 3."""

from __future__ import annotations

from gymnasium import spaces

from cfisac.agents.base import ALGORITHMS, Agent, AgentConfig
from cfisac.agents.ppo import A2CAgent, PPOAgent
from cfisac.agents.sac import SACAgent
from cfisac.agents.td3 import DDPGAgent, TD3Agent

_REGISTRY: dict[str, type[Agent]] = {
    "sac": SACAgent,
    "td3": TD3Agent,
    "ddpg": DDPGAgent,
    "ppo": PPOAgent,
    "a2c": A2CAgent,
}


def make_agent(
    config: AgentConfig,
    obs_space: spaces.Box,
    action_space: spaces.Box,
    seed: int = 0,
    buffer_capacity: int | None = None,
) -> Agent:
    """Instantiate the agent selected by ``config.algo``."""
    return _REGISTRY[config.algo](obs_space, action_space, config, seed, buffer_capacity)


__all__ = [
    "ALGORITHMS",
    "A2CAgent",
    "Agent",
    "AgentConfig",
    "DDPGAgent",
    "PPOAgent",
    "SACAgent",
    "TD3Agent",
    "make_agent",
]
