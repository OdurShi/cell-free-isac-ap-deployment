from __future__ import annotations

import itertools

import numpy as np
import pytest

from cfisac.optimize import (
    DeterministicObjective,
    best_from_table,
    count_grid_deployments,
    differential_evolution_search,
    exhaustive_search,
    exhaustive_table,
    local_search,
    optimise,
)
from cfisac.scenario import ScenarioConfig, archived_scenario


def test_count_grid_deployments():
    assert count_grid_deployments(ScenarioConfig(n_tx=1, n_rx=1)) == 36 * 36
    assert count_grid_deployments(ScenarioConfig(n_tx=2, n_rx=2)) == 666 * 666


@pytest.mark.parametrize(
    "cfg",
    [
        archived_scenario(1, 1),
        ScenarioConfig(n_tx=1, n_rx=2),
        ScenarioConfig(n_tx=2, n_rx=1, aggregation="max-min"),
    ],
)
def test_exhaustive_table_matches_direct_evaluation(cfg, rng):
    obj = DeterministicObjective(cfg, n_ue_snapshots=64, seed=0)
    table = exhaustive_table(obj)
    assert table.comm_sum.shape == (len(table.tx_sets), len(table.rx_sets))
    for _ in range(25):
        i, j = rng.integers(len(table.tx_sets)), rng.integers(len(table.rx_sets))
        direct = obj.breakdown(obj.sites[table.tx_sets[i]], obj.sites[table.rx_sets[j]])
        fast = obj.breakdown_sites(table.tx_sets[i], table.rx_sets[j])
        for b in (direct, fast):
            assert b.comm_sum == pytest.approx(table.comm_sum[i, j], rel=1e-9)
            assert b.comm_min == pytest.approx(table.comm_min[i, j], rel=1e-9)
            assert b.sens_mean == pytest.approx(table.sens_mean[i, j], rel=1e-9, abs=1e-10)
            assert b.sens_min == pytest.approx(table.sens_min[i, j], rel=1e-9, abs=1e-10)


def test_exhaustive_search_is_the_brute_force_optimum():
    cfg = archived_scenario(1, 1)
    obj = DeterministicObjective(cfg, n_ue_snapshots=64, seed=0)
    result = exhaustive_search(obj)
    best = max(
        cfg.objective_value(obj.breakdown(obj.sites[[i]], obj.sites[[j]]))
        for i, j in itertools.product(range(36), repeat=2)
    )
    assert result.value == pytest.approx(best, rel=1e-12)
    np.testing.assert_allclose(result.tx, [[40.0, 10.0]])
    np.testing.assert_allclose(result.rx, [[40.0, 10.0]])
    assert result.method == "exhaustive" and result.evaluations == 36 * 36


def test_candidate_values_match_single_evaluations(rng):
    cfg = ScenarioConfig(n_tx=2, n_rx=2, aggregation="max-min")
    obj = DeterministicObjective(cfg, n_ue_snapshots=32, seed=1)
    idx = rng.integers(0, 36, size=4)
    for position in range(4):
        values = obj.candidate_values(idx, position)
        assert values.shape == (36,)
        for site in (0, 13, 35):
            trial = idx.copy()
            trial[position] = site
            expected = cfg.objective_value(obj.breakdown_sites(trial[:2], trial[2:]))
            assert values[site] == pytest.approx(expected, rel=1e-9, abs=1e-12)


@pytest.mark.parametrize(
    "cfg",
    [
        archived_scenario(1, 1),
        ScenarioConfig(n_tx=1, n_rx=2),
        archived_scenario(1, 2, "max-min"),
    ],
)
def test_local_search_reaches_the_exhaustive_optimum(cfg):
    obj = DeterministicObjective(cfg, n_ue_snapshots=64, seed=0)
    reference = exhaustive_search(obj)
    result = local_search(obj, n_starts=24, seed=0)
    assert result.value == pytest.approx(reference.value, rel=1e-9)
    again = local_search(DeterministicObjective(cfg, 64, 0), n_starts=24, seed=0)
    np.testing.assert_array_equal(result.tx, again.tx)
    np.testing.assert_array_equal(result.rx, again.rx)


def test_table_optimum_per_objective():
    cfg = archived_scenario(1, 1)
    obj = DeterministicObjective(cfg, n_ue_snapshots=64, seed=0)
    table = exhaustive_table(obj)
    sens = best_from_table(obj, table, "max-sum", "sensing")
    comm = best_from_table(obj, table, "max-sum", "communication")
    isac = best_from_table(obj, table, "max-sum", "isac")
    assert sens.breakdown.sens_mean >= isac.breakdown.sens_mean >= 0
    assert comm.breakdown.comm_sum >= isac.breakdown.comm_sum
    assert isac.value >= sens.breakdown.comm_sum * sens.breakdown.sens_mean
    assert isac.value >= comm.breakdown.comm_sum * comm.breakdown.sens_mean


def test_differential_evolution_on_continuous_region():
    cfg = archived_scenario(1, 1).with_updates(grid_step=None)
    obj = DeterministicObjective(cfg, n_ue_snapshots=32, seed=0)
    result = differential_evolution_search(obj, seed=0, maxiter=25, popsize=10)
    grid = exhaustive_search(DeterministicObjective(archived_scenario(1, 1), 32, 0))
    assert result.method == "differential-evolution"
    assert result.value >= 0.95 * grid.value
    for positions in (result.tx, result.rx):
        assert np.all((positions >= 0) & (positions <= 50))


def test_optimise_dispatch():
    assert optimise(archived_scenario(1, 1), n_ue_snapshots=32).method == "exhaustive"
    local = optimise(archived_scenario(1, 1), n_ue_snapshots=32, exhaustive_limit=10, n_starts=4)
    assert local.method == "local-search"
    with pytest.raises(ValueError):
        optimise(archived_scenario(1, 1), method="annealing")
    payload = local.to_dict()
    assert {"tx", "rx", "value", "comm_sum", "comm_min", "sens_mean", "sens_min"} <= set(payload)
