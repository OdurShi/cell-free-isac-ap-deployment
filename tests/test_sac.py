"""SAC (Section 3.2): losses of Eqs. (22)-(24), update mechanics, learning smoke test."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from cfisac.agents import AgentConfig, SACAgent, make_agent
from cfisac.env import APDeploymentEnv
from cfisac.scenario import ScenarioConfig
from tests.helpers import ACT_SPACE, OBS_SPACE, TARGET, run_on_quadratic


def _agent(**overrides) -> SACAgent:
    env = APDeploymentEnv(ScenarioConfig())
    cfg = AgentConfig(algo="sac", batch_size=32, learning_starts=0, **overrides)
    agent = make_agent(cfg, env.observation_space, env.action_space, seed=0, buffer_capacity=500)
    assert isinstance(agent, SACAgent)
    obs, _ = env.reset(seed=0)
    for _ in range(64):
        action = agent.act(obs)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        agent.store(obs, action, reward, next_obs, terminated, truncated)
        obs = next_obs
    return agent


def test_table1_defaults():
    cfg = AgentConfig()
    assert cfg.algo == "sac"
    assert cfg.hidden_sizes == (64, 32)
    assert cfg.learning_rate == 1e-5
    assert cfg.buffer_size == 2**21
    assert cfg.batch_size == 2**9
    assert cfg.gamma == 0.98
    assert cfg.tau == 0.005


def test_action_shapes_and_target_entropy():
    torch.manual_seed(0)
    agent = _agent()
    assert agent.target_entropy == -8.0  # -dim(A), below Eq. (24)
    assert agent.omega.item() == pytest.approx(math.exp(-1.0))
    obs = np.zeros(8, dtype=np.float32)
    for deterministic in (False, True):
        action = agent.act(obs, deterministic=deterministic)
        assert action.shape == (8,) and action.dtype == np.float32
        assert np.all(np.abs(action) <= 1.0)
    np.testing.assert_array_equal(agent.act(obs, True), agent.act(obs, True))


def test_critic_loss_is_soft_bellman_residual_eq23():
    torch.manual_seed(0)
    agent = _agent()
    batch = agent.buffer.sample(32)
    torch.manual_seed(5)
    loss = agent._critic_loss(batch)
    # Manual evaluation with the same random draw of a_{i+1}.
    torch.manual_seed(5)
    with torch.no_grad():
        next_action, next_log_prob = agent.actor.sample(batch["next_obs"])
        q1_t, q2_t = agent.critic_target(batch["next_obs"], next_action)
        target = batch["reward"] + agent.config.gamma * (
            torch.min(q1_t, q2_t) - agent.omega * next_log_prob
        )
        q1, q2 = agent.critic(batch["obs"], batch["action"])
        expected = 0.5 * ((q1 - target) ** 2).mean() + 0.5 * ((q2 - target) ** 2).mean()
    assert loss.shape == ()
    assert loss.item() == pytest.approx(expected.item(), rel=1e-5)


def test_actor_and_temperature_losses_eq22_eq24():
    torch.manual_seed(0)
    agent = _agent()
    batch = agent.buffer.sample(32)
    torch.manual_seed(3)
    actor_loss, log_prob = agent._actor_loss(batch)
    torch.manual_seed(3)
    action, lp = agent.actor.sample(batch["obs"])
    expected = (agent.omega.detach() * lp - agent.critic.min_q(batch["obs"], action)).mean()
    assert actor_loss.item() == pytest.approx(expected.item(), rel=1e-5)
    assert log_prob.shape == (32,) and not log_prob.requires_grad

    omega_loss = agent._temperature_loss(log_prob)
    expected = (-agent.omega * log_prob - agent.omega * agent.target_entropy).mean()
    assert omega_loss.item() == pytest.approx(expected.item())
    # Entropy above the target (-8) -> the temperature must decrease, and vice versa.
    for fake_log_prob, sign in ((torch.full((32,), 2.0), 1.0), (torch.full((32,), 20.0), -1.0)):
        agent.log_omega.grad = None
        agent._temperature_loss(fake_log_prob).backward()
        assert math.copysign(1.0, agent.log_omega.grad.item()) == sign


def test_update_step_changes_all_networks_and_soft_updates_target():
    torch.manual_seed(0)
    agent = _agent(learning_rate=1e-3)
    before = {
        name: [p.detach().clone() for p in module.parameters()]
        for name, module in agent.modules().items()
    }
    omega_before = agent.omega.item()
    logs = agent.update_step()
    assert set(logs) == {"critic_loss", "actor_loss", "omega", "entropy"}
    assert all(math.isfinite(v) for v in logs.values())
    for name in ("actor", "critic"):
        assert any(
            not torch.equal(old, new)
            for old, new in zip(before[name], agent.modules()[name].parameters(), strict=True)
        )
    assert agent.omega.item() != omega_before
    tau = agent.config.tau
    for old, new, online in zip(
        before["critic_target"], agent.critic_target.parameters(), agent.critic.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(new, tau * online + (1 - tau) * old)
    assert all(p.requires_grad for p in agent.critic.parameters())
    assert not any(p.requires_grad for p in agent.critic_target.parameters())


def test_update_waits_for_enough_data_and_counts_steps():
    agent = make_agent(
        AgentConfig(algo="sac", batch_size=16, learning_starts=40, horizon_len=8),
        OBS_SPACE, ACT_SPACE, seed=0, buffer_capacity=100,
    )
    obs = np.zeros(2, dtype=np.float32)
    for _ in range(24):
        agent.store(obs, agent.act(obs), 0.0, obs, False, False)
    assert agent.update() == {}
    for _ in range(24):
        agent.store(obs, agent.act(obs), 0.0, obs, False, False)
    assert "critic_loss" in agent.update()
    assert agent.total_steps == 48


def test_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(0)
    agent = _agent(learning_rate=1e-3)
    agent.update_step()
    path = tmp_path / "sac.pt"
    agent.save(path)
    state = torch.load(path, map_location="cpu", weights_only=True)  # tensors only
    assert {"actor", "critic", "critic_target", "log_omega"} <= set(state)
    env = APDeploymentEnv(ScenarioConfig())
    clone = make_agent(agent.config, env.observation_space, env.action_space, seed=1)
    clone.load(path)
    obs = env.reset(seed=3)[0]
    np.testing.assert_allclose(clone.act(obs, True), agent.act(obs, True), rtol=1e-6)
    assert clone.omega.item() == pytest.approx(agent.omega.item())


def test_sac_learns_trivial_objective():
    torch.manual_seed(0)
    cfg = AgentConfig(
        algo="sac", hidden_sizes=(32, 32), learning_rate=3e-3, batch_size=64, gamma=0.0,
        horizon_len=16, learning_starts=64, init_temperature=0.05, tau=0.05,
    )
    agent = run_on_quadratic(cfg, steps=1200)
    action = agent.act(np.zeros(2, dtype=np.float32), deterministic=True)
    assert np.abs(action - TARGET).max() < 0.15
