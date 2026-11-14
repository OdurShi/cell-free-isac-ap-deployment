"""Non-learning optimisers used to cross-check the objective values reachable in a scenario.

They maximise the same objective as the RL agents (Eq. 17 or a Table 3 variant) but on a
*fixed* set of UE snapshots, so that every deployment has one deterministic score:

* :func:`exhaustive_search` -- every deployment on the candidate grid (small ``M + N``),
* :func:`local_search` -- multi-start coordinate ascent on the candidate grid,
* :func:`differential_evolution_search` -- continuous AP coordinates (SciPy).
"""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import differential_evolution

from cfisac.data import EVAL_SEED, load_ue_positions
from cfisac.metrics import (
    ObjectiveBreakdown,
    combine_fim_terms,
    evaluate_deployment,
    unit_directions,
)
from cfisac.scenario import ScenarioConfig

logger = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class SearchResult:
    """Best deployment found by an optimiser."""

    tx: FloatArray
    rx: FloatArray
    value: float
    breakdown: ObjectiveBreakdown
    method: str
    evaluations: int

    def to_dict(self) -> dict:
        return {
            "tx": self.tx.tolist(),
            "rx": self.rx.tolist(),
            "value": self.value,
            "comm_sum": self.breakdown.comm_sum,
            "comm_min": self.breakdown.comm_min,
            "sens_mean": self.breakdown.sens_mean,
            "sens_min": self.breakdown.sens_min,
            "method": self.method,
            "evaluations": self.evaluations,
        }


@dataclass(frozen=True)
class BreakdownTable:
    """The four Table 3 quantities for every (transmit multiset, receive multiset) pair."""

    tx_sets: IntArray  # [I, M] site indices
    rx_sets: IntArray  # [J, N] site indices
    comm_sum: FloatArray  # [I, J]
    comm_min: FloatArray
    sens_mean: FloatArray
    sens_min: FloatArray

    def objective(self, aggregation: str, objective: str, weight: float) -> FloatArray:
        """Objective value of every deployment, shape ``[I, J]``."""
        comm = self.comm_sum if aggregation == "max-sum" else self.comm_min
        sens = self.sens_mean if aggregation == "max-sum" else self.sens_min
        if objective == "isac":
            return comm * sens
        if objective == "sensing":
            return sens
        if objective == "communication":
            return comm
        if objective == "weighted":
            return weight * comm + (1.0 - weight) * sens
        raise ValueError(objective)


class DeterministicObjective:
    """Scores deployments on a fixed sample of UE snapshots (common random numbers).

    The sensing factors are exact for the sampled trajectory.  The communication factors are
    averages over ``n_ue_snapshots`` UE snapshots drawn once from the scenario's Gaussian UE
    distribution -- the same quantity an RL evaluation estimates with fresh draws at every
    step, which makes optimiser and RL numbers directly comparable.

    Args:
        scenario: the scenario (its ``aggregation`` / ``objective`` select the default score).
        n_ue_snapshots: number of UE snapshots in the fixed sample.
        seed: seed of the fixed sample.  For the seed of the shipped evaluation split
            (``cfisac.data.EVAL_SEED``) the first ``n_ue_snapshots`` rows of
            ``data/ue_positions_eval.npy`` are used when that file is available and large
            enough; otherwise the sample is drawn from the scenario's UE distribution.
        data_dir: directory of the shipped data (``None``: default location).
    """

    def __init__(
        self,
        scenario: ScenarioConfig,
        n_ue_snapshots: int = 4096,
        seed: int = 0,
        data_dir: str | None = None,
    ):
        self.scenario = scenario
        self.targets = scenario.trajectory()
        pool = load_ue_positions("eval", scenario, data_dir) if seed == EVAL_SEED else None
        if pool is not None and len(pool) >= n_ue_snapshots:
            self.ues = np.asarray(pool[:n_ue_snapshots], dtype=np.float64)  # [S, K, 2]
        else:
            self.ues = scenario.sample_ues(np.random.default_rng(seed), n_ue_snapshots)
        self.evaluations = 0
        self._sites: FloatArray | None = None
        self._site_dirs: FloatArray | None = None
        self._site_inv: FloatArray | None = None

    # ------------------------------------------------------- direct evaluation
    def breakdown(self, tx: FloatArray, rx: FloatArray) -> ObjectiveBreakdown:
        """Factors of Eq. (17) for arbitrary (continuous) AP positions."""
        cfg = self.scenario
        self.evaluations += 1
        return evaluate_deployment(
            tx,
            rx,
            self.targets,
            self.ues,
            sensing_metric=cfg.sensing_metric,
            log_base=cfg.log_base,
            one_plus_snr=cfg.one_plus_snr,
            min_distance=cfg.min_distance,
        )

    def value(self, tx: FloatArray, rx: FloatArray) -> float:
        """Scalar objective selected by the scenario."""
        return self.scenario.objective_value(self.breakdown(tx, rx))

    # ------------------------------------------------- candidate-site fast path
    @property
    def sites(self) -> FloatArray:
        """Candidate AP sites ``[G, 2]`` (requires a discretised action space)."""
        if self._sites is None:
            cfg = self.scenario
            self._sites = cfg.candidate_sites()
            self._site_dirs = unit_directions(self._sites, self.targets, cfg.min_distance)
            diff = self.ues[None, :, :, :] - self._sites[:, None, None, :]  # [G, S, K, 2]
            dist_sq = np.maximum((diff * diff).sum(-1), cfg.min_distance**2)
            self._site_inv = (1.0 / dist_sq).transpose(0, 2, 1)  # [G, K, S]
        return self._sites

    def _rate(self, snr: FloatArray) -> FloatArray:
        cfg = self.scenario
        arg = 1.0 + snr if cfg.one_plus_snr else snr
        return np.log(arg) / np.log(cfg.log_base)

    def _sensing_sites(self, tx_idx: IntArray, rx_idx: FloatArray) -> FloatArray:
        assert self._site_dirs is not None
        ut = self._site_dirs[:, tx_idx, :]
        ur = self._site_dirs[:, rx_idx, :]
        pair = ut[:, :, None, :] + ur[:, None, :, :]
        c, s = pair[..., 0], pair[..., 1]
        return combine_fim_terms(
            (c * c).sum((1, 2)), (s * s).sum((1, 2)), (c * s).sum((1, 2)),
            self.scenario.sensing_metric,
        )

    def breakdown_sites(self, tx_idx: IntArray, rx_idx: IntArray) -> ObjectiveBreakdown:
        """Same as :meth:`breakdown` for APs given as indices into :attr:`sites`."""
        _ = self.sites
        assert self._site_inv is not None
        self.evaluations += 1
        det = self._sensing_sites(np.asarray(tx_idx), np.asarray(rx_idx))
        idx = np.concatenate([np.asarray(tx_idx), np.asarray(rx_idx)])
        per_user = self._rate(self._site_inv[idx].sum(axis=0)).mean(axis=-1)  # [K]
        return ObjectiveBreakdown(
            float(per_user.sum()), float(per_user.min()), float(det.mean()), float(det.min())
        )

    def candidate_values(self, idx: IntArray, position: int) -> FloatArray:
        """Objective for every candidate site of AP ``position`` with the other APs fixed.

        ``idx`` holds the site indices of all ``M + N`` APs (transmit APs first).
        """
        cfg = self.scenario
        sites = self.sites
        assert self._site_inv is not None and self._site_dirs is not None
        n_sites = len(sites)
        others = np.delete(idx, position)
        base = self._site_inv[others].sum(axis=0)  # [K, S]
        per_user = self._rate(base[None] + self._site_inv).mean(axis=-1)  # [G, K]
        comm_sum, comm_min = per_user.sum(axis=1), per_user.min(axis=1)

        dirs = self._site_dirs  # [Q, G, 2]
        m = cfg.n_tx
        is_tx = position < m
        same = np.delete(idx[:m] if is_tx else idx[m:], position if is_tx else position - m)
        other_role = idx[m:] if is_tx else idx[:m]
        uo = dirs[:, other_role, :]  # [Q, O, 2]
        # Pairs that do not involve the moving AP.
        fixed = dirs[:, same, None, :] + uo[:, None, :, :]
        fa = (fixed[..., 0] ** 2).sum((1, 2))
        fb = (fixed[..., 1] ** 2).sum((1, 2))
        fc = (fixed[..., 0] * fixed[..., 1]).sum((1, 2))
        # Pairs of the moving AP (at each candidate site) with the other role.
        moving = dirs[:, :, None, :] + uo[:, None, :, :]  # [Q, G, O, 2]
        a = fa[:, None] + (moving[..., 0] ** 2).sum(-1)
        b = fb[:, None] + (moving[..., 1] ** 2).sum(-1)
        c = fc[:, None] + (moving[..., 0] * moving[..., 1]).sum(-1)
        det = combine_fim_terms(a, b, c, cfg.sensing_metric)  # [Q, G]
        sens_mean, sens_min = det.mean(axis=0), det.min(axis=0)
        self.evaluations += n_sites

        comm = comm_sum if cfg.aggregation == "max-sum" else comm_min
        sens = sens_mean if cfg.aggregation == "max-sum" else sens_min
        if cfg.objective == "isac":
            return comm * sens
        if cfg.objective == "sensing":
            return sens
        if cfg.objective == "communication":
            return comm
        return cfg.weight * comm + (1.0 - cfg.weight) * sens

    def site_indices(self, tx: FloatArray, rx: FloatArray) -> IntArray:
        """Indices into :attr:`sites` of a deployment that lies on the candidate grid."""
        positions = np.concatenate([np.asarray(tx, float).reshape(-1, 2),
                                    np.asarray(rx, float).reshape(-1, 2)])
        dist = np.linalg.norm(positions[:, None, :] - self.sites[None, :, :], axis=-1)
        idx = dist.argmin(axis=1)
        if not np.allclose(dist[np.arange(len(idx)), idx], 0.0, atol=1e-9):
            raise ValueError("deployment is not on the candidate grid")
        return idx

    def result_from_sites(self, idx: IntArray, method: str) -> SearchResult:
        """Package a deployment given as site indices."""
        m = self.scenario.n_tx
        breakdown = self.breakdown_sites(idx[:m], idx[m:])
        return SearchResult(
            tx=self.sites[idx[:m]].copy(),
            rx=self.sites[idx[m:]].copy(),
            value=self.scenario.objective_value(breakdown),
            breakdown=breakdown,
            method=method,
            evaluations=self.evaluations,
        )


def _multisets(n_sites: int, size: int) -> IntArray:
    return np.array(list(itertools.combinations_with_replacement(range(n_sites), size)), dtype=int)


def count_grid_deployments(scenario: ScenarioConfig) -> int:
    """Number of distinct deployments on the candidate grid (AP order within a role ignored)."""
    g = len(scenario.candidate_sites())
    from math import comb

    return comb(g + scenario.n_tx - 1, scenario.n_tx) * comb(g + scenario.n_rx - 1, scenario.n_rx)


def exhaustive_table(objective: DeterministicObjective) -> BreakdownTable:
    """Evaluate every deployment on the candidate grid.

    Uses the separability of the double sums of Eq. (8): with ``T1 = sum_m alpha^t_m`` and
    ``T2 = sum_m alpha^t_m (x) alpha^t_m`` (and ``R1``, ``R2`` likewise),
    ``sum_m sum_n (alpha^t_m + alpha^r_n)(alpha^t_m + alpha^r_n)^T
    = N T2 + M R2 + T1 R1^T + R1 T1^T``.
    """
    cfg = objective.scenario
    sites = objective.sites
    assert objective._site_dirs is not None and objective._site_inv is not None
    dirs, inv = objective._site_dirs, objective._site_inv
    tx_sets = _multisets(len(sites), cfg.n_tx)
    rx_sets = _multisets(len(sites), cfg.n_rx)

    def moments(sets: IntArray) -> tuple[FloatArray, ...]:
        u = dirs[:, sets, :]  # [Q, I, M, 2]
        x, y = u[..., 0], u[..., 1]
        return x.sum(-1), y.sum(-1), (x * x).sum(-1), (y * y).sum(-1), (x * y).sum(-1)

    t1x, t1y, t2x, t2y, t2xy = moments(tx_sets)  # each [Q, I]
    r1x, r1y, r2x, r2y, r2xy = moments(rx_sets)  # each [Q, J]
    m, n = cfg.n_tx, cfg.n_rx
    shape = (len(tx_sets), len(rx_sets))
    sens_mean, sens_min = np.zeros(shape), np.zeros(shape)
    comm_sum, comm_min = np.zeros(shape), np.zeros(shape)
    rx_inv = inv[rx_sets].sum(axis=1)  # [J, K, S]
    symmetric = m == n
    for i in range(len(tx_sets)):
        a = n * t2x[:, i, None] + m * r2x + 2.0 * t1x[:, i, None] * r1x
        b = n * t2y[:, i, None] + m * r2y + 2.0 * t1y[:, i, None] * r1y
        c = n * t2xy[:, i, None] + m * r2xy + t1x[:, i, None] * r1y + t1y[:, i, None] * r1x
        det = combine_fim_terms(a, b, c, cfg.sensing_metric)  # [Q, J]
        sens_mean[i], sens_min[i] = det.mean(axis=0), det.min(axis=0)
        # The SNR depends only on the union of all APs, so for M == N row i equals column i.
        start = i if symmetric else 0
        tx_inv = inv[tx_sets[i]].sum(axis=0)  # [K, S]
        per_user = objective._rate(tx_inv[None] + rx_inv[start:]).mean(axis=-1)  # [J', K]
        comm_sum[i, start:] = per_user.sum(axis=1)
        comm_min[i, start:] = per_user.min(axis=1)
        if symmetric:
            comm_sum[start:, i] = comm_sum[i, start:]
            comm_min[start:, i] = comm_min[i, start:]
    objective.evaluations += shape[0] * shape[1]
    return BreakdownTable(tx_sets, rx_sets, comm_sum, comm_min, sens_mean, sens_min)


def best_from_table(
    objective: DeterministicObjective,
    table: BreakdownTable,
    aggregation: str | None = None,
    objective_name: str | None = None,
    weight: float | None = None,
) -> SearchResult:
    """Arg-max of an objective over an exhaustive table (first maximiser wins ties)."""
    cfg = objective.scenario
    values = table.objective(
        aggregation or cfg.aggregation,
        objective_name or cfg.objective,
        cfg.weight if weight is None else weight,
    )
    i, j = np.unravel_index(int(np.argmax(values)), values.shape)
    breakdown = ObjectiveBreakdown(
        float(table.comm_sum[i, j]),
        float(table.comm_min[i, j]),
        float(table.sens_mean[i, j]),
        float(table.sens_min[i, j]),
    )
    return SearchResult(
        tx=objective.sites[table.tx_sets[i]].copy(),
        rx=objective.sites[table.rx_sets[j]].copy(),
        value=float(values[i, j]),
        breakdown=breakdown,
        method="exhaustive",
        evaluations=values.size,
    )


def exhaustive_search(objective: DeterministicObjective) -> SearchResult:
    """Global optimum on the candidate grid by enumeration."""
    return best_from_table(objective, exhaustive_table(objective))


def local_search(
    objective: DeterministicObjective,
    n_starts: int = 32,
    seed: int = 0,
    max_sweeps: int = 50,
    colocated_starts: bool = True,
    extra_starts: Sequence[IntArray] = (),
) -> SearchResult:
    """Multi-start coordinate ascent on the candidate grid.

    From a starting deployment, each AP in turn is moved to its best candidate site with all
    other APs fixed; sweeps repeat until no AP moves.  Starts are ``n_starts`` random
    deployments plus, if ``colocated_starts`` is set, one deployment per candidate site with
    all APs on that site (single-AP moves rarely assemble such clusters from a random start),
    plus any ``extra_starts`` given as arrays of ``M + N`` site indices.
    The best run is returned.  Ties are broken towards the lowest site index, so results are
    deterministic for a given seed.  The result is a local optimum, not a certified global one.
    """
    rng = np.random.default_rng(seed)
    n_sites = len(objective.sites)
    n_aps = objective.scenario.n_aps
    starts = [rng.integers(0, n_sites, size=n_aps) for _ in range(n_starts)]
    if colocated_starts:
        starts += [np.full(n_aps, site, dtype=int) for site in range(n_sites)]
    starts += [np.array(start, dtype=int) for start in extra_starts]
    best_idx: IntArray | None = None
    best_value = -np.inf
    for idx in starts:
        value = float(objective.candidate_values(idx, 0)[idx[0]])
        for _ in range(max_sweeps):
            moved = False
            for position in range(n_aps):
                values = objective.candidate_values(idx, position)
                choice = int(np.argmax(values))
                if values[choice] > value + 1e-12 * max(1.0, abs(value)):
                    idx[position] = choice
                    value = float(values[choice])
                    moved = True
            if not moved:
                break
        if value > best_value:
            best_value, best_idx = value, idx.copy()
    assert best_idx is not None
    return objective.result_from_sites(best_idx, "local-search")


def differential_evolution_search(
    objective: DeterministicObjective, seed: int = 0, maxiter: int = 200, popsize: int = 20
) -> SearchResult:
    """Continuous-coordinate optimum with SciPy's differential evolution."""
    cfg = objective.scenario
    bounds = [(cfg.x_min, cfg.x_max), (cfg.y_min, cfg.y_max)] * cfg.n_aps

    def negative(x: FloatArray) -> float:
        pos = x.reshape(cfg.n_aps, 2)
        return -objective.value(pos[: cfg.n_tx], pos[cfg.n_tx :])

    res = differential_evolution(
        negative, bounds, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-8, polish=False
    )
    pos = res.x.reshape(cfg.n_aps, 2)
    tx, rx = pos[: cfg.n_tx], pos[cfg.n_tx :]
    breakdown = objective.breakdown(tx, rx)
    return SearchResult(
        tx, rx, cfg.objective_value(breakdown), breakdown, "differential-evolution",
        objective.evaluations,
    )


def optimise(
    scenario: ScenarioConfig,
    method: str = "auto",
    n_ue_snapshots: int = 4096,
    n_starts: int = 32,
    seed: int = 0,
    exhaustive_limit: int = 500_000,
) -> SearchResult:
    """Optimise a scenario's objective without reinforcement learning.

    ``method="auto"`` enumerates the candidate grid when it has at most ``exhaustive_limit``
    deployments, falls back to :func:`local_search` otherwise, and uses differential
    evolution when the action space is continuous (``grid_step=None``).
    """
    objective = DeterministicObjective(scenario, n_ue_snapshots, seed)
    if method == "auto":
        if scenario.grid_step is None:
            method = "de"
        elif count_grid_deployments(scenario) <= exhaustive_limit:
            method = "exhaustive"
        else:
            method = "local"
    if method == "exhaustive":
        return exhaustive_search(objective)
    if method == "local":
        return local_search(objective, n_starts, seed)
    if method == "de":
        return differential_evolution_search(objective, seed)
    raise ValueError(f"unknown method {method!r}")
