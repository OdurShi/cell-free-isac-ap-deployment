"""The shipped scenario data (data/) and the loaders that use it."""

from __future__ import annotations

import json

import numpy as np
import pytest

from cfisac import data as cfdata
from cfisac.env import APDeploymentEnv
from cfisac.metrics import evaluate_deployment
from cfisac.optimize import DeterministicObjective
from cfisac.scenario import ScenarioConfig, archived_scenario
from cfisac.training import TrainConfig, train
from tests.conftest import REPO_ROOT

DATA = REPO_ROOT / "data"


@pytest.fixture
def no_data(monkeypatch, tmp_path):
    """Point the loaders at an empty directory."""
    monkeypatch.setenv(cfdata.DATA_ENV_VAR, str(tmp_path))
    return tmp_path


def test_shipped_files_match_manifest():
    assert cfdata.default_data_dir() == DATA
    manifest = cfdata.load_manifest()
    assert set(manifest["files"]) == {
        "ue_positions_train.npy", "ue_positions_eval.npy", "trajectory_eq8.csv",
        "trajectory_archived.csv", "candidate_sites.csv", "ue_centers.csv",
    }
    assert all(cfdata.verify().values())
    for name, entry in manifest["files"].items():
        assert (DATA / name).stat().st_size == entry["bytes"]
    readme = (DATA / "README.md").read_text(encoding="utf-8")
    for entry in manifest["files"].values():
        assert entry["sha256"] in readme


def test_ue_position_splits():
    cfg = ScenarioConfig()
    train_pos = cfdata.load_ue_positions("train", cfg)
    eval_pos = cfdata.load_ue_positions("eval", cfg)
    assert train_pos.shape == (409_600, 3, 2) and train_pos.dtype == np.float32
    assert eval_pos.shape == (32_768, 3, 2) and eval_pos.dtype == np.float64
    for positions in (train_pos, eval_pos):
        np.testing.assert_allclose(positions.mean(axis=0), np.array(cfg.ue_centers), atol=0.03)
        np.testing.assert_allclose(positions.var(axis=0), cfg.ue_variance, rtol=0.05)
    with pytest.raises(ValueError):
        cfdata.load_ue_positions("test", cfg)
    # Data generated for another UE distribution are not used.
    assert cfdata.load_ue_positions("train", cfg.with_updates(ue_variance=3.0)) is None
    assert cfdata.load_ue_positions("eval", archived_scenario(4, 4)) is not None


def test_trajectory_and_site_files_equal_the_configuration():
    # Tolerance of a few ulp: cos / sin may differ in the last bit between platforms.
    np.testing.assert_allclose(
        cfdata.load_trajectory("eq8"), ScenarioConfig().trajectory(), rtol=1e-13, atol=1e-12
    )
    np.testing.assert_allclose(
        cfdata.load_trajectory("archived"), archived_scenario().trajectory(), rtol=1e-13,
        atol=1e-12,
    )
    sites = np.loadtxt(DATA / "candidate_sites.csv", delimiter=",", skiprows=1)[:, 1:]
    np.testing.assert_array_equal(sites, ScenarioConfig().candidate_sites())
    centers = np.loadtxt(DATA / "ue_centers.csv", delimiter=",", skiprows=1)[:, 1:]
    np.testing.assert_array_equal(centers, np.array(ScenarioConfig().ue_centers))


def test_generator_reproduces_the_shipped_layout(tmp_path):
    manifest = cfdata.generate(tmp_path)
    shipped = cfdata.load_manifest()
    assert manifest["ue_distribution"] == shipped["ue_distribution"]
    for name in ("candidate_sites.csv", "ue_centers.csv"):
        assert manifest["files"][name]["sha256"] == shipped["files"][name]["sha256"]
    for family in ("eq8", "archived"):
        np.testing.assert_allclose(
            cfdata.load_trajectory(family, tmp_path), cfdata.load_trajectory(family),
            rtol=1e-13, atol=1e-12,
        )
    for name in ("ue_positions_train.npy", "ue_positions_eval.npy"):
        assert manifest["files"][name]["shape"] == shipped["files"][name]["shape"]
        assert manifest["files"][name]["seed"] == shipped["files"][name]["seed"]
    assert all(cfdata.verify(tmp_path).values())
    assert json.loads((tmp_path / "manifest.json").read_text())["files"].keys() == (
        shipped["files"].keys()
    )


def test_env_uses_blocks_of_the_pool():
    cfg = ScenarioConfig()
    pool = cfdata.load_ue_positions("train", cfg)[:250]  # 10 blocks of Q = 25
    env = APDeploymentEnv(cfg, pool)
    obs, _ = env.reset(seed=3)
    blocks = pool.reshape(10, 25, 3, 2).astype(np.float64)
    rng = np.random.default_rng(0)
    for step in range(30):
        matches = [i for i in range(10) if np.array_equal(blocks[i], env._ues)]
        assert len(matches) == 1
        np.testing.assert_allclose(obs[:-2], env._ues[step % 25].ravel(), rtol=1e-6)
        action = rng.uniform(-1, 1, cfg.action_dim)
        ues = env._ues.copy()
        obs, reward, *_ = env.step(action)
        tx, rx = cfg.action_to_positions(action)
        assert reward == pytest.approx(evaluate_deployment(tx, rx, cfg.trajectory(), ues).isac(
            cfg.aggregation))
    with pytest.raises(ValueError):
        APDeploymentEnv(cfg, pool[:10])
    with pytest.raises(ValueError):
        APDeploymentEnv(cfg, np.zeros((100, 2, 2)))


def test_objective_uses_eval_file_and_matches_generation():
    cfg = archived_scenario(2, 2)
    from_file = DeterministicObjective(cfg, n_ue_snapshots=512, seed=cfdata.EVAL_SEED)
    np.testing.assert_array_equal(
        from_file.ues, cfdata.load_ue_positions("eval", cfg)[:512]
    )
    # Another seed, or more snapshots than the file holds, are drawn from the distribution.
    other = DeterministicObjective(cfg, n_ue_snapshots=512, seed=7)
    assert not np.array_equal(other.ues, from_file.ues)
    assert DeterministicObjective(cfg, n_ue_snapshots=40_000).ues.shape == (40_000, 3, 2)


def test_fallback_to_on_the_fly_draws_without_data(no_data, tmp_path):
    cfg = archived_scenario(2, 2).with_updates(episode_length=16)
    assert cfdata.default_data_dir() == no_data
    assert cfdata.load_manifest() is None
    assert cfdata.load_ue_positions("train", cfg) is None
    assert cfdata.load_trajectory("eq8") is None
    with pytest.raises(FileNotFoundError):
        cfdata.verify()
    assert DeterministicObjective(cfg, 64).ues.shape == (64, 3, 2)
    from cfisac.agents import AgentConfig

    agent = AgentConfig(algo="sac", batch_size=16, horizon_len=16, learning_starts=16)
    train(cfg, agent, TrainConfig(total_steps=32, eval_interval=16, eval_episodes=1),
          tmp_path / "run")
    hyper = json.loads((tmp_path / "run" / "hyperparameter.json").read_text())
    assert hyper["ue_positions"] == {"train": "drawn on the fly", "eval": "drawn on the fly"}


def test_training_reads_the_data_files_by_default(tmp_path):
    from cfisac.agents import AgentConfig

    cfg = archived_scenario(2, 2).with_updates(episode_length=16)
    agent = AgentConfig(algo="sac", batch_size=16, horizon_len=16, learning_starts=16)
    train(cfg, agent, TrainConfig(total_steps=32, eval_interval=16, eval_episodes=1),
          tmp_path / "data")
    hyper = json.loads((tmp_path / "data" / "hyperparameter.json").read_text())
    assert hyper["ue_positions"] == {"train": "data file", "eval": "data file"}
    train(cfg, agent,
          TrainConfig(total_steps=32, eval_interval=16, eval_episodes=1, online_ues=True),
          tmp_path / "online")
    hyper = json.loads((tmp_path / "online" / "hyperparameter.json").read_text())
    assert hyper["ue_positions"]["train"] == "drawn on the fly"
