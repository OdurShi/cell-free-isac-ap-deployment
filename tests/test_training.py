"""End-to-end: training loop, recorder format, checkpoints, determinism."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from cfisac.agents import AgentConfig, make_agent
from cfisac.analysis import load_best_locations, load_recorder
from cfisac.env import APDeploymentEnv
from cfisac.scenario import archived_scenario
from cfisac.training import (
    BEST_LOCATION_FILE,
    RECORDER_COLUMNS,
    RECORDER_FILE,
    TrainConfig,
    evaluate_policy,
    run_directory,
    train,
)

SCENARIO = archived_scenario(2, 2).with_updates(episode_length=32)


def _config(algo: str) -> AgentConfig:
    return AgentConfig(algo=algo, batch_size=32, horizon_len=32, learning_starts=32,
                       learning_rate=1e-3)


@pytest.mark.parametrize("algo", ["sac", "ppo"])
def test_train_writes_recorder_and_checkpoints(tmp_path, algo):
    train_cfg = TrainConfig(total_steps=160, eval_interval=64, eval_episodes=2, seed=3)
    summary = train(SCENARIO, _config(algo), train_cfg, tmp_path)
    assert summary["steps"] == 160

    rec = load_recorder(tmp_path / RECORDER_FILE)
    assert tuple(k for k in rec if k != "index") == RECORDER_COLUMNS
    # Evaluations after the first horizon, then every eval_interval, and at the end.
    np.testing.assert_array_equal(rec["steps"], [32, 96, 160])
    assert np.all(np.isfinite(rec["rewards"]))
    # Cumulative reward equals cumulative ISAC objective for the default objective.
    np.testing.assert_allclose(rec["rewards"], rec["rewards_isac"])
    assert np.all(np.diff(rec["best_rewards"]) >= 0)
    np.testing.assert_allclose(
        rec["best_rewards"][-1], rec["rewards"].max() / SCENARIO.episode_length
    )

    locations = load_best_locations(tmp_path / BEST_LOCATION_FILE)
    assert locations.shape == (3, 4, 2)
    assert np.all(locations % 10 == 0)

    best = json.loads((tmp_path / "best_deployment.json").read_text())
    np.testing.assert_allclose(np.array(best["tx"] + best["rx"]), locations[-1])
    assert best["reward"] == pytest.approx(rec["best_rewards"][-1])
    hyper = json.loads((tmp_path / "hyperparameter.json").read_text())
    assert hyper["agent"]["algo"] == algo and hyper["scenario"]["sensing_metric"] == "archived"

    env = APDeploymentEnv(SCENARIO)
    agent = make_agent(_config(algo), env.observation_space, env.action_space, seed=0,
                       buffer_capacity=10)
    for name in ("checkpoint_last.pt", "checkpoint_best.pt"):
        state = torch.load(tmp_path / name, map_location="cpu", weights_only=True)
        agent.load_state_dict(state)


def test_training_is_deterministic_for_a_seed(tmp_path):
    cfg = TrainConfig(total_steps=96, eval_interval=32, eval_episodes=1, seed=11)
    train(SCENARIO, _config("sac"), cfg, tmp_path / "a")
    train(SCENARIO, _config("sac"), cfg, tmp_path / "b")
    rec_a = load_recorder(tmp_path / "a" / RECORDER_FILE)
    rec_b = load_recorder(tmp_path / "b" / RECORDER_FILE)
    for column in RECORDER_COLUMNS:
        if column != "times":
            np.testing.assert_array_equal(rec_a[column], rec_b[column])


def test_evaluate_policy_reports_modal_deployment():
    env = APDeploymentEnv(SCENARIO)
    env.reset(seed=0)

    class Fixed:
        def act(self, obs, deterministic=False):
            return SCENARIO.positions_to_action([[40, 10], [40, 10]], [[40, 10], [40, 0]])

    result = evaluate_policy(Fixed(), env, episodes=3)
    assert result.episode_reward.shape == (3,) and result.steps_per_episode == 32
    np.testing.assert_allclose(result.tx, [[40, 10], [40, 10]])
    np.testing.assert_allclose(result.rx, [[40, 10], [40, 0]])
    # Sensing does not depend on the UE draws: 47.3749 per step in the archive for this layout.
    np.testing.assert_allclose(result.episode_sens / 32, 47.3749, rtol=1e-5)
    np.testing.assert_allclose(result.episode_reward, result.episode_isac)


def test_run_directory_layout():
    path = run_directory("root", AgentConfig(algo="td3"), SCENARIO, seed=128)
    assert path.as_posix() == "root/td3/2t2r/seed-128"
