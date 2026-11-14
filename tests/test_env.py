from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from cfisac.env import APDeploymentEnv
from cfisac.metrics import evaluate_deployment
from cfisac.scenario import ScenarioConfig, archived_scenario


@pytest.mark.parametrize(
    "config",
    [ScenarioConfig(), archived_scenario(1, 1), ScenarioConfig(n_tx=4, n_rx=4, grid_step=None)],
)
def test_gymnasium_api(config):
    env = APDeploymentEnv(config)
    check_env(env, skip_render_check=True)
    assert env.observation_space.shape == (config.obs_dim,)
    assert env.action_space.shape == (config.action_dim,)


def test_observation_is_state_of_eq18():
    cfg = ScenarioConfig()
    env = APDeploymentEnv(cfg)
    obs, _ = env.reset(seed=0)
    traj = cfg.trajectory()
    np.testing.assert_allclose(obs[-2:], traj[0], rtol=1e-6)
    np.testing.assert_allclose(obs[:-2], env._ues[0].ravel(), rtol=1e-6)
    for step in range(1, 30):
        obs, *_ = env.step(env.action_space.sample())
        np.testing.assert_allclose(obs[-2:], traj[step % 25], rtol=1e-6)


@pytest.mark.parametrize("aggregation", ["max-sum", "max-min"])
@pytest.mark.parametrize("metric", ["eq8", "archived"])
def test_reward_is_objective_of_eq17(aggregation, metric):
    cfg = ScenarioConfig(aggregation=aggregation, sensing_metric=metric)
    env = APDeploymentEnv(cfg)
    env.reset(seed=4)
    rng = np.random.default_rng(0)
    for _ in range(5):
        action = rng.uniform(-1, 1, size=cfg.action_dim)
        ues = env._ues.copy()  # the snapshots the reward of this step is computed on
        _, reward, terminated, truncated, info = env.step(action)
        tx, rx = cfg.action_to_positions(action)
        b = evaluate_deployment(tx, rx, cfg.trajectory(), ues, sensing_metric=metric)
        assert reward == pytest.approx(b.isac(aggregation))
        assert info["isac"] == pytest.approx(reward)
        assert info["comm"] * info["sens"] == pytest.approx(reward)
        np.testing.assert_allclose(info["tx"], tx)
        np.testing.assert_allclose(info["rx"], rx)
        assert not terminated and not truncated
        assert np.all(info["tx"] % 10 == 0) and np.all(info["rx"] % 10 == 0)


def test_table3_objectives_and_reward_scale():
    action = np.array([0.6, -0.6, -1.0, -1.0, 0.2, 0.2, 1.0, 1.0])
    rewards = {}
    for objective in ("isac", "sensing", "communication", "weighted"):
        env = APDeploymentEnv(ScenarioConfig(objective=objective, weight=0.3, reward_scale=2.0))
        env.reset(seed=1)
        _, reward, _, _, info = env.step(action)
        rewards[objective] = (reward, info)
    r, info = rewards["sensing"]
    assert r == pytest.approx(2.0 * info["sens"])
    r, info = rewards["communication"]
    assert r == pytest.approx(2.0 * info["comm"])
    r, info = rewards["weighted"]
    assert r == pytest.approx(2.0 * (0.3 * info["comm"] + 0.7 * info["sens"]))
    r, info = rewards["isac"]
    assert r == pytest.approx(2.0 * info["comm"] * info["sens"])


def test_episode_truncation_and_seeding():
    cfg = ScenarioConfig(episode_length=7)
    env_a, env_b = APDeploymentEnv(cfg), APDeploymentEnv(cfg)
    obs_a, _ = env_a.reset(seed=9)
    obs_b, _ = env_b.reset(seed=9)
    np.testing.assert_array_equal(obs_a, obs_b)
    action = np.zeros(cfg.action_dim, dtype=np.float32)
    for step in range(7):
        obs_a, r_a, term, trunc, _ = env_a.step(action)
        obs_b, r_b, _, _, _ = env_b.step(action)
        np.testing.assert_array_equal(obs_a, obs_b)
        assert r_a == r_b
        assert term is False
        assert trunc is (step == 6)
    obs_c, _ = env_a.reset(seed=10)
    assert not np.array_equal(obs_c, env_b.reset(seed=9)[0])


def test_eq8_reward_is_zero_for_one_tx_one_rx():
    env = APDeploymentEnv(ScenarioConfig(n_tx=1, n_rx=1))
    env.reset(seed=0)
    for _ in range(5):
        _, reward, *_ = env.step(env.action_space.sample())
        assert abs(reward) < 1e-12
