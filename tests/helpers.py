"""Shared helpers for agent tests: a trivial one-step objective."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

from cfisac.agents import Agent, AgentConfig, make_agent

OBS_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
ACT_SPACE = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
TARGET = np.array([0.5, -0.5], dtype=np.float32)


def quadratic_reward(action: np.ndarray) -> float:
    """Trivial objective with its maximum at ``TARGET``."""
    return -float(((np.asarray(action) - TARGET) ** 2).sum())


def run_on_quadratic(config: AgentConfig, steps: int, seed: int = 0) -> Agent:
    """Train an agent on the one-step quadratic objective (constant observation)."""
    agent = make_agent(config, OBS_SPACE, ACT_SPACE, seed=seed, buffer_capacity=steps)
    obs = np.zeros(2, dtype=np.float32)
    for step in range(1, steps + 1):
        action = agent.act(obs)
        agent.store(obs, action, quadratic_reward(action), obs, terminated=False, truncated=False)
        if step % config.horizon_len == 0:
            agent.update()
    return agent
