"""The ``archived`` scenario reproduces values recorded by the original study.

The pairs below are copied from the archived ``recorder_my.csv`` /
``recorder_best_location.csv`` files (columns ``best_rewards_s`` and the matching best
location).  When the ``experiments/`` tree is present the same check runs on every recorded
deployment.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

from cfisac.cli import figure_check
from cfisac.metrics import fim_determinant
from cfisac.optimize import DeterministicObjective
from cfisac.paper_values import TABLE3
from cfisac.scenario import archived_scenario

# (transmit APs, receive APs, archived best_rewards_s)
RECORDED = [
    ([[20, 20]], [[20, 20]], 1.939783238157992),
    ([[40, 10]], [[40, 0]], 4.12900),
    ([[40, 10]], [[40, 10]], 4.014915560161114),
    ([[40, 10]], [[50, 0]], 4.62098),
    ([[40, 20]], [[30, 10]], 2.32863),
    ([[40, 10], [40, 10]], [[40, 10], [40, 10]], 45.7491),
    ([[40, 0], [40, 10]], [[40, 10], [40, 0]], 48.5750),
    ([[40, 10], [50, 0]], [[50, 0], [50, 0]], 57.7765),
    ([[0, 0], [50, 0]], [[0, 50], [40, 10]], 17.2639),
]


@pytest.mark.parametrize("tx, rx, recorded", RECORDED)
def test_recorded_sensing_values(tx, rx, recorded):
    cfg = archived_scenario(len(tx), len(rx))
    value = fim_determinant(tx, rx, cfg.trajectory(), "archived").mean()
    assert value == pytest.approx(recorded, rel=2e-5)


def test_eq8_cannot_produce_the_recorded_values():
    # With Eq. (8) as printed the 1T1R sensing value is identically zero, while the archive
    # records 4.0149 for both APs at (40, 10).
    cfg = archived_scenario(1, 1)
    value = fim_determinant([[40, 10]], [[40, 10]], cfg.trajectory(), "eq8").mean()
    assert abs(value) < 1e-12


def test_all_recorded_deployments(archive_dir):
    pairs: dict[str, float] = {}
    for recorder in sorted(archive_dir.glob("*/*/seed-*/recorder_my.csv")):
        with open(recorder, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        locations = (recorder.parent / "recorder_best_location.csv").read_text().splitlines()
        assert len(rows) == len(locations)
        for row, location in zip(rows, locations, strict=True):
            pairs[location] = float(row["best_rewards_s"])
    assert len(pairs) >= 70
    worst = 0.0
    for location, recorded in pairs.items():
        aps = np.array([float(v) for v in location.split(",")]).reshape(-1, 2)
        m = len(aps) // 2
        cfg = archived_scenario(m, m)
        value = fim_determinant(aps[:m], aps[m:], cfg.trajectory(), "archived").mean()
        worst = max(worst, abs(value - recorded) / recorded)
    assert worst < 1e-6


def test_recorded_rate_level_of_converged_sac():
    # Archived SAC 2T2R runs end with all APs at (40, 10) and a mean per-step communication
    # factor of 0.512 (rewards_c / 256).  Base-10 logarithm and UE variance 2 reproduce it.
    cfg = archived_scenario(2, 2)
    site = np.array([[40.0, 10.0], [40.0, 10.0]])
    b = DeterministicObjective(cfg, n_ue_snapshots=20000, seed=3).breakdown(site, site)
    assert b.comm_sum == pytest.approx(0.512, rel=0.03)
    assert b.sens_mean * b.comm_sum * 256 == pytest.approx(6000.0, rel=0.03)  # Fig. 3(a) plateau


def test_table3_sensing_entries_from_figure_deployments():
    rows = {row["deployment"]: row for row in figure_check(n_ue_snapshots=512)}
    assert rows["fig4a_sac_max_sum"]["sens_mean"] == pytest.approx(
        TABLE3["isac"]["sens_mean"], abs=0.006
    )
    assert rows["fig4e_comm_only_max_sum"]["sens_mean"] == pytest.approx(
        TABLE3["communication"]["sens_mean"], abs=0.006
    )
    assert rows["fig4f_sensing_only"]["sens_mean"] == pytest.approx(
        TABLE3["sensing"]["sens_mean"], abs=0.006
    )
    assert rows["fig4f_sensing_only"]["sens_min"] == pytest.approx(
        TABLE3["sensing"]["sens_min"], abs=0.006
    )
    assert rows["fig5e_comm_only_max_min"]["sens_min"] == pytest.approx(
        TABLE3["communication"]["sens_min"], abs=0.006
    )
    # Far-field rates do not depend on the UE sample: 8 APs at the corner (0, 0).
    assert rows["fig4f_sensing_only"]["comm_sum"] == pytest.approx(
        TABLE3["sensing"]["comm_sum"], abs=6e-4
    )
    assert rows["fig4f_sensing_only"]["comm_min"] == pytest.approx(
        TABLE3["sensing"]["comm_min"], abs=6e-5
    )
