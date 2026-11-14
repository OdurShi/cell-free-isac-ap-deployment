"""DDPG, TD3, A2C and PPO baselines behind the common agent interface."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cfisac.agents import (
    ALGORITHMS,
    A2CAgent,
    AgentConfig,
    DDPGAgent,
    PPOAgent,
    TD3Agent,
    make_agent,
)
from cfisac.agents.networks import QNetwork, TwinQNetwork
from tests.helpers import ACT_SPACE, OBS_SPACE, TARGET, quadratic_reward, run_on_quadratic


def test_registry_and_config_validation(tmp_path):
    assert set(ALGORITHMS) == {"sac", "td3", "ddpg", "ppo", "a2c"}
    kinds = {"td3": TD3Agent, "ddpg": DDPGAgent, "ppo": PPOAgent, "a2c": A2CAgent}
    for algo, cls in kinds.items():
        agent = make_agent(AgentConfig(algo=algo), OBS_SPACE, ACT_SPACE, buffer_capacity=10)
        assert type(agent) is cls
        assert agent.on_policy is (algo in ("ppo", "a2c"))
    with pytest.raises(ValueError):
        AgentConfig(algo="dqn")
    with pytest.raises(ValueError, match="unknown agent fields"):
        AgentConfig.from_dict({"algo": "sac", "lr": 1.0})
    path = tmp_path / "agent.yaml"
    path.write_text("algo: td3\nhidden_sizes: [16, 8]\nbatch_size: 32\n", encoding="utf-8")
    cfg = AgentConfig.from_yaml(path)
    assert cfg.algo == "td3" and cfg.hidden_sizes == (16, 8) and cfg.batch_size == 32


def test_ddpg_has_single_critic_and_td3_twin():
    ddpg = make_agent(AgentConfig(algo="ddpg"), OBS_SPACE, ACT_SPACE, buffer_capacity=10)
    td3 = make_agent(AgentConfig(algo="td3"), OBS_SPACE, ACT_SPACE, buffer_capacity=10)
    assert isinstance(ddpg.critic, QNetwork) and isinstance(td3.critic, TwinQNetwork)
    assert ddpg.policy_delay == 1 and td3.policy_delay == 2


@pytest.mark.parametrize("algo", ["td3", "ddpg", "ppo", "a2c"])
def test_update_runs_and_checkpoints(algo, tmp_path):
    torch.manual_seed(0)
    cfg = AgentConfig(algo=algo, batch_size=32, horizon_len=32, learning_starts=16,
                      learning_rate=1e-3)
    agent = make_agent(cfg, OBS_SPACE, ACT_SPACE, seed=0, buffer_capacity=200)
    obs = np.zeros(2, dtype=np.float32)
    before = [p.detach().clone() for p in agent.modules()["actor"].parameters()]
    logs = {}
    for step in range(1, 97):
        action = agent.act(obs)
        assert action.shape == (2,) and np.all(np.abs(action) <= 1.0)
        agent.store(obs, action, quadratic_reward(action), obs, False, step % 48 == 0)
        if step % cfg.horizon_len == 0:
            logs = agent.update()
    assert "critic_loss" in logs and all(math.isfinite(v) for v in logs.values())
    after = list(agent.modules()["actor"].parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    path = tmp_path / f"{algo}.pt"
    agent.save(path)
    clone = make_agent(cfg, OBS_SPACE, ACT_SPACE, seed=5, buffer_capacity=200)
    clone.load(path)
    np.testing.assert_allclose(clone.act(obs, True), agent.act(obs, True), rtol=1e-6)


def test_on_policy_store_requires_act():
    agent = make_agent(AgentConfig(algo="ppo", horizon_len=4), OBS_SPACE, ACT_SPACE)
    obs = np.zeros(2, dtype=np.float32)
    with pytest.raises(RuntimeError):
        agent.store(obs, np.zeros(2), 0.0, obs, False, False)
    assert agent.update() == {}  # rollout not full yet


@pytest.mark.parametrize("algo", ["td3", "ddpg"])
def test_deterministic_policy_gradient_learns_trivial_objective(algo):
    torch.manual_seed(0)
    cfg = AgentConfig(
        algo=algo, hidden_sizes=(32, 32), learning_rate=3e-3, batch_size=64, gamma=0.0,
        horizon_len=16, learning_starts=64, tau=0.05, explore_noise_std=0.2,
    )
    agent = run_on_quadratic(cfg, steps=1200)
    action = agent.act(np.zeros(2, dtype=np.float32), deterministic=True)
    assert np.abs(action - TARGET).max() < 0.15


@pytest.mark.parametrize("algo", ["ppo", "a2c"])
def test_on_policy_improves_trivial_objective(algo):
    torch.manual_seed(0)
    cfg = AgentConfig(
        algo=algo, hidden_sizes=(32, 32), learning_rate=3e-3, batch_size=64, gamma=0.0,
        horizon_len=64, repeat_times=8.0, entropy_coef=0.0,
    )
    obs = np.zeros(2, dtype=np.float32)
    start = make_agent(cfg, OBS_SPACE, ACT_SPACE, seed=0).act(obs, deterministic=True)
    agent = run_on_quadratic(cfg, steps=64 * 60)
    end = agent.act(obs, deterministic=True)
    assert quadratic_reward(end) > quadratic_reward(start)
    assert np.abs(end - TARGET).max() < 0.25
