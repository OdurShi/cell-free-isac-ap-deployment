"""Smoke tests of the ``cfisac`` command line."""

from __future__ import annotations

import json
import subprocess
import sys

import pytest

from cfisac.cli import main
from tests.conftest import REPO_ROOT


def test_help_runs_as_module():
    out = subprocess.run(
        [sys.executable, "-m", "cfisac.cli", "--help"], capture_output=True, text=True, check=True
    )
    assert "train" in out.stdout and "optimize" in out.stdout


def test_train_curves_deployment_pipeline(tmp_path, capsys):
    scenario = tmp_path / "scenario.yaml"
    scenario.write_text(
        "n_tx: 1\nn_rx: 1\nsensing_metric: archived\ntarget_center: [25.05, 25.05]\n"
        "episode_length: 16\n",
        encoding="utf-8",
    )
    agent = tmp_path / "agent.yaml"
    agent.write_text(
        "algo: sac\nbatch_size: 16\nhorizon_len: 16\nlearning_starts: 16\n", encoding="utf-8"
    )
    runs = tmp_path / "runs"
    for seed in ("1", "2"):
        code = main([
            "train", "--scenario", str(scenario), "--agent", str(agent), "--seed", seed,
            "--total-steps", "64", "--eval-interval", "32", "--eval-episodes", "1",
            "--out", str(runs),
        ])
        assert code == 0
    capsys.readouterr()
    run_dir = runs / "sac" / "1t1r" / "seed-1"
    assert (run_dir / "recorder.csv").exists() and (run_dir / "best_deployment.json").exists()

    figure = tmp_path / "curves.png"
    assert main(["curves", "--runs", str(runs), "--setting", "1t1r", "--out", str(figure)]) == 0
    assert figure.exists() and figure.with_suffix(".json").exists()
    assert main(["curves", "--runs", str(runs), "--setting", "9t9r", "--out", str(figure)]) == 1

    dep = tmp_path / "dep.png"
    assert main(["deployment", "--scenario", str(scenario), "--run", str(run_dir),
                 "--out", str(dep)]) == 0
    assert dep.exists()


def test_optimize_evaluate_and_tables(tmp_path, capsys):
    out = tmp_path / "opt.json"
    code = main(["optimize", "--scenario", str(REPO_ROOT / "configs/archived/maxsum_1t1r.yaml"),
                 "--ue-snapshots", "64", "--out", str(out)])
    assert code == 0
    result = json.loads(out.read_text())["result"]
    assert result["tx"] == [[40.0, 10.0]] and result["rx"] == [[40.0, 10.0]]

    capsys.readouterr()
    assert main(["evaluate", "--scenario", str(REPO_ROOT / "configs/eq8/maxsum_2t2r.yaml"),
                 "--tx", "0,25;50,25", "--rx", "25,0;25,50", "--ue-snapshots", "64"]) == 0
    values = json.loads(capsys.readouterr().out)
    assert values["sens_mean"] > 0 and values["objective"] == pytest.approx(
        values["comm_sum"] * values["sens_mean"]
    )
    with pytest.raises(SystemExit):
        main(["evaluate", "--tx", "0,0"])

    tables = tmp_path / "tables"
    assert main(["tables", "--sensing-metric", "archived", "--counts", "1", "--table3-count", "1",
                 "--ue-snapshots", "64", "--out", str(tables)]) == 0
    data = json.loads((tables / "tables_archived.json").read_text())
    assert data["table2"][0]["n_aps"] == 2 and set(data["table3"]) == {
        "isac", "sensing", "communication", "weighted",
    }
    assert "Table 3 layout" in (tables / "tables_archived.md").read_text()


def test_check_table3(tmp_path, capsys):
    out = tmp_path / "check.md"
    assert main(["check-table3", "--ue-snapshots", "128", "--out", str(out)]) == 0
    text = out.read_text()
    assert "688.97" in text and "842.92" in text and "188.40" in text


def test_curves_on_archive(tmp_path, archive_dir, capsys):
    figure = tmp_path / "fig3a.png"
    code = main(["curves", "--runs", str(archive_dir), "--setting", "2t2r",
                 "--max-steps", "3e6", "--out", str(figure)])
    assert code == 0 and figure.exists()
    summary = json.loads(figure.with_suffix(".json").read_text())
    assert summary["sac"]["mean"] > max(summary[a]["mean"] for a in ("ddpg", "td3", "a2c", "ppo"))


def test_agent_configs_match_archived_hyperparameters(archive_dir):
    from cfisac.agents import AgentConfig

    for algo in ("sac", "td3", "ddpg", "a2c", "ppo"):
        cfg = AgentConfig.from_yaml(REPO_ROOT / f"configs/agents/{algo}.yaml")
        for path in sorted(archive_dir.glob(f"{algo}/*/seed-*/hyperparameter.json")):
            recorded = json.loads(path.read_text())
            assert recorded["learning_rate"] == cfg.learning_rate
            assert recorded["batch_size"] == cfg.batch_size
            assert tuple(recorded["net_dims"]) == cfg.hidden_sizes
            assert recorded["gamma"] == cfg.gamma
            assert recorded["soft_update_tau"] == cfg.tau
            assert recorded["clip_grad_norm"] == cfg.clip_grad_norm
            assert recorded["reward_scale"] == cfg.reward_scale
            assert recorded["horizon_len"] == cfg.horizon_len
            if recorded["buffer_size"] is not None:
                assert recorded["buffer_size"] == cfg.buffer_size


def test_configs_load():
    from cfisac.agents import AgentConfig
    from cfisac.scenario import ScenarioConfig, archived_scenario

    scenario_files = sorted((REPO_ROOT / "configs").glob("*/m*.yaml"))
    assert len(scenario_files) == 16
    for path in scenario_files:
        cfg = ScenarioConfig.from_yaml(path)
        assert cfg.sensing_metric == path.parent.name
        assert f"{cfg.n_tx}t{cfg.n_rx}r" in path.name
        assert cfg.aggregation.replace("-", "") in path.name
    assert ScenarioConfig.from_yaml(REPO_ROOT / "configs/archived/maxsum_2t2r.yaml") == (
        archived_scenario(2, 2)
    )
    assert ScenarioConfig.from_yaml(REPO_ROOT / "configs/eq8/maxsum_2t2r.yaml") == ScenarioConfig()
    for algo in ("sac", "td3", "ddpg", "a2c", "ppo"):
        cfg = AgentConfig.from_yaml(REPO_ROOT / f"configs/agents/{algo}.yaml")
        assert cfg.algo == algo and cfg.learning_rate == 1e-5 and cfg.gamma == 0.98
        assert cfg.batch_size == (1024 if algo in ("a2c", "ppo") else 512)
    assert AgentConfig.from_yaml(REPO_ROOT / "configs/agents/sac.yaml") == AgentConfig()
    fast = AgentConfig.from_yaml(REPO_ROOT / "configs/agents/sac_fast.yaml")
    assert fast == AgentConfig(learning_rate=3e-4)  # differs from Table 1 in the rate only
