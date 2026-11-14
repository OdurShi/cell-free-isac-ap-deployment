from __future__ import annotations

import csv

import numpy as np
import pytest

from cfisac.analysis import (
    aggregate_runs,
    discover_runs,
    final_performance,
    load_best_locations,
    load_recorder,
)
from cfisac.plotting import plot_deployment, plot_learning_curves
from cfisac.scenario import archived_scenario
from cfisac.training import RECORDER_COLUMNS


def _write_run(root, algo, seed, steps, rewards, name="recorder.csv"):
    run = root / algo / "2t2r" / f"seed-{seed}"
    run.mkdir(parents=True)
    with open(run / name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("", *RECORDER_COLUMNS))
        for i, (s, r) in enumerate(zip(steps, rewards, strict=True)):
            writer.writerow([i, s, i, r] + [0.0] * (len(RECORDER_COLUMNS) - 3))
    return run


def test_load_and_aggregate_synthetic_runs(tmp_path):
    _write_run(tmp_path, "sac", 1, [0, 100, 200, 300], [0.0, 10.0, 20.0, 30.0])
    _write_run(tmp_path, "sac", 2, [0, 150, 300, 450], [0.0, 30.0, 60.0, 90.0], "recorder_my.csv")
    _write_run(tmp_path, "td3", 1, [0, 300], [5.0, 5.0])
    runs = discover_runs(tmp_path)
    assert [(r.algo, r.setting, r.seed) for r in runs] == [
        ("sac", "2t2r", 1), ("sac", "2t2r", 2), ("td3", "2t2r", 1),
    ]
    rec = load_recorder(runs[0].path)
    assert "index" in rec and rec["steps"].tolist() == [0, 100, 200, 300]

    curve = aggregate_runs(runs[:2], "rewards", n_points=4)
    np.testing.assert_allclose(curve.steps, [0, 100, 200, 300])  # shortest run ends at 300
    np.testing.assert_allclose(curve.mean, [0.0, 15.0, 30.0, 45.0])
    np.testing.assert_allclose(curve.low, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(curve.high, [0.0, 20.0, 40.0, 60.0])
    assert curve.n_seeds == 2
    capped = aggregate_runs(runs[:2], "rewards", max_steps=200, n_points=3)
    np.testing.assert_allclose(capped.steps, [0, 100, 200])

    final = final_performance(runs[:2], "rewards", last_fraction=0.25)
    assert final["per_seed"] == {1: 30.0, 2: 90.0} and final["mean"] == 60.0
    with pytest.raises(ValueError):
        aggregate_runs([])

    out = plot_learning_curves(
        {"sac": curve, "td3": aggregate_runs(runs[2:], n_points=4)}, tmp_path / "fig" / "c.png"
    )
    assert out.exists() and out.stat().st_size > 2000


def test_deployment_plot(tmp_path):
    out = plot_deployment(
        archived_scenario(2, 2), [[40, 10], [40, 10]], [[40, 10], [0, 0]], tmp_path / "d.png"
    )
    assert out.exists() and out.stat().st_size > 2000


def test_archived_recorders_load(archive_dir):
    runs = discover_runs(archive_dir)
    assert len(runs) == 20
    assert {r.algo for r in runs} == {"a2c", "ddpg", "ppo", "sac", "td3"}
    assert sorted(r.seed for r in runs if r.algo == "sac" and r.setting == "2t2r") == [1, 128, 256]
    sac = [r for r in runs if r.algo == "sac" and r.setting == "2t2r"]
    rec = load_recorder(sac[0].path)
    assert set(RECORDER_COLUMNS) <= set(rec)
    assert rec["steps"][0] == 256 and rec["steps"][1] - rec["steps"][0] == 20224
    # Fig. 3(a): the archived SAC runs plateau near 6000 for all three seeds.
    final = final_performance(sac)
    assert final["min"] > 5900 and final["max"] < 6100
    locations = load_best_locations(sac[0].path.parent / "recorder_best_location.csv")
    assert locations.shape[1:] == (4, 2)
    np.testing.assert_allclose(locations[-1], [[40, 10]] * 4)
