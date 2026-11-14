from __future__ import annotations

import numpy as np
import pytest

from cfisac.scenario import ScenarioConfig, archived_scenario


def test_default_dimensions():
    cfg = ScenarioConfig()
    assert (cfg.n_aps, cfg.n_users, cfg.obs_dim, cfg.action_dim) == (4, 3, 8, 8)
    cfg = ScenarioConfig(n_tx=1, n_rx=1)
    assert (cfg.obs_dim, cfg.action_dim) == (8, 4)


def test_trajectory_is_a_sampled_circle():
    cfg = ScenarioConfig()
    traj = cfg.trajectory()
    assert traj.shape == (25, 2)
    np.testing.assert_allclose(np.linalg.norm(traj - np.array(cfg.target_center), axis=1), 10.0)
    np.testing.assert_allclose(traj[0], [35.0, 25.0])
    np.testing.assert_allclose(traj[0], traj[-1], atol=1e-12)  # endpoint repeats the start
    np.testing.assert_allclose(traj[6], [25.0, 35.0], atol=1e-12)  # 24 distinct points, 15 deg
    open_traj = cfg.with_updates(trajectory_endpoint=False, n_trajectory_points=24).trajectory()
    np.testing.assert_allclose(open_traj, traj[:-1], atol=1e-12)


def test_candidate_sites_and_snapping():
    cfg = ScenarioConfig()
    sites = cfg.candidate_sites()
    assert sites.shape == (36, 2)
    assert {tuple(s) for s in sites} == {(x, y) for x in range(0, 51, 10) for y in range(0, 51, 10)}
    np.testing.assert_allclose(cfg.snap([[12.4, 37.6], [-3.0, 80.0]]), [[10.0, 40.0], [0.0, 50.0]])
    free = cfg.with_updates(grid_step=None)
    np.testing.assert_allclose(free.snap([[12.4, 37.6], [-3.0, 80.0]]), [[12.4, 37.6], [0.0, 50.0]])
    with pytest.raises(ValueError):
        free.candidate_sites()


def test_action_mapping_layout_and_roundtrip():
    cfg = ScenarioConfig(n_tx=1, n_rx=2)
    tx, rx = cfg.action_to_positions([-1.0, 1.0, 0.6, -0.6, 0.0, 0.21])
    np.testing.assert_allclose(tx, [[0.0, 50.0]])
    np.testing.assert_allclose(rx, [[40.0, 10.0], [20.0, 30.0]])  # 25 -> 20 (round half to even)
    action = cfg.positions_to_action(tx, rx)
    tx2, rx2 = cfg.action_to_positions(action)
    np.testing.assert_allclose(tx2, tx)
    np.testing.assert_allclose(rx2, rx)
    # Out-of-range actions are clipped to the region of Eq. (16).
    tx3, _ = cfg.action_to_positions([5.0, -7.0, 0, 0, 0, 0])
    np.testing.assert_allclose(tx3, [[50.0, 0.0]])


def test_ue_sampling_statistics():
    cfg = ScenarioConfig()
    ues = cfg.sample_ues(np.random.default_rng(0), 40000)
    assert ues.shape == (40000, 3, 2)
    np.testing.assert_allclose(ues.mean(axis=0), np.array(cfg.ue_centers), atol=0.03)
    np.testing.assert_allclose(ues.var(axis=0), 2.0, rtol=0.05)
    assert cfg.sample_ues(np.random.default_rng(0)).shape == (25, 3, 2)
    low, high = cfg.obs_bounds()
    assert low.shape == (8,) and np.all(low == -10.0) and np.all(high == 60.0)


def test_yaml_roundtrip_and_validation(tmp_path):
    import yaml

    cfg = archived_scenario(4, 4, "max-min")
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(cfg.to_dict()), encoding="utf-8")
    assert ScenarioConfig.from_yaml(path) == cfg
    with pytest.raises(ValueError, match="unknown scenario fields"):
        ScenarioConfig.from_dict({"n_tx": 2, "typo": 1})
    for bad in (
        {"n_tx": 0},
        {"sensing_metric": "eq9"},
        {"aggregation": "max"},
        {"objective": "both"},
        {"grid_step": 0.0},
        {"x_max": -1.0},
        {"log_base": 1.0},
    ):
        with pytest.raises(ValueError):
            ScenarioConfig(**bad)


def test_archived_scenario_differs_only_in_metric_and_centre():
    base, arch = ScenarioConfig().to_dict(), archived_scenario().to_dict()
    changed = {k for k in base if base[k] != arch[k]}
    assert changed == {"sensing_metric", "target_center"}
