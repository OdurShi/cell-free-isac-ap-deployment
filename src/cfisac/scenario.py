"""Scenario of Section 4: region, target trajectory, UE distribution and objective settings.

Every constant that the paper leaves unstated is a field of :class:`ScenarioConfig`.  The
defaults and their justification are listed in ``docs/implementation_notes.md``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray

from cfisac.metrics import (
    AGGREGATIONS,
    DEFAULT_MIN_DISTANCE,
    OBJECTIVES,
    SENSING_METRICS,
    ObjectiveBreakdown,
    evaluate_deployment,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ScenarioConfig:
    """Cell-free ISAC deployment scenario.

    Attributes:
        n_tx: number ``M`` of transmit APs.
        n_rx: number ``N`` of receive APs.
        x_min, x_max, y_min, y_max: deployment region of Eq. (16), in metres.
        grid_step: spacing of the candidate AP sites (the paper discretises the action
            space); ``None`` keeps AP coordinates continuous.
        target_center: centre of the circular target trajectory.
        target_radius: radius of the trajectory.
        n_trajectory_points: number ``Q`` of sampled trajectory points.
        trajectory_endpoint: if ``True`` the samples are ``linspace(0, 2*pi, Q)`` including
            both ends (first point repeated), otherwise ``Q`` distinct equispaced points.
        ue_centers: mean position of each of the ``K`` UEs.
        ue_variance: per-coordinate variance of the Gaussian UE distribution.
        log_base: base of the logarithm of Eq. (15).
        one_plus_snr: ``True`` uses ``log(1 + SNR)``; ``False`` the literal last line of Eq. (15).
        sensing_metric: ``"eq8"`` (paper) or ``"archived"`` (original logs).
        aggregation: ``"max-sum"`` or ``"max-min"`` problem of Eq. (17).
        objective: ``"isac"`` (Eq. 17) or one of the Table 3 comparison objectives.
        weight: communication weight of the ``"weighted"`` objective.
        reward_scale: constant multiplied onto the reward (Eq. 20 differs from Eq. 17 only by
            constant factors of ``Q``).
        episode_length: number of environment steps per episode.
        obs_margin: margin added around the region for the observation bounds; UE draws are
            clipped to it.
        min_distance: lower clip of every AP-UE and AP-target distance.
    """

    n_tx: int = 2
    n_rx: int = 2
    x_min: float = 0.0
    x_max: float = 50.0
    y_min: float = 0.0
    y_max: float = 50.0
    grid_step: float | None = 10.0
    target_center: tuple[float, float] = (25.0, 25.0)
    target_radius: float = 10.0
    n_trajectory_points: int = 25
    trajectory_endpoint: bool = True
    ue_centers: tuple[tuple[float, float], ...] = ((25.0, 45.0), (10.0, 15.0), (40.0, 10.0))
    ue_variance: float = 2.0
    log_base: float = 10.0
    one_plus_snr: bool = True
    sensing_metric: str = "eq8"
    aggregation: str = "max-sum"
    objective: str = "isac"
    weight: float = 0.5
    reward_scale: float = 1.0
    episode_length: int = 256
    obs_margin: float = 10.0
    min_distance: float = DEFAULT_MIN_DISTANCE

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_center", tuple(float(v) for v in self.target_center))
        object.__setattr__(
            self, "ue_centers", tuple(tuple(float(v) for v in c) for c in self.ue_centers)
        )
        if self.n_tx < 1 or self.n_rx < 1:
            raise ValueError("at least one transmit and one receive AP are required")
        if not (self.x_max > self.x_min and self.y_max > self.y_min):
            raise ValueError("empty deployment region")
        if self.grid_step is not None and self.grid_step <= 0:
            raise ValueError("grid_step must be positive or None")
        if self.n_trajectory_points < 1:
            raise ValueError("n_trajectory_points must be >= 1")
        if self.sensing_metric not in SENSING_METRICS:
            raise ValueError(f"sensing_metric must be one of {SENSING_METRICS}")
        if self.aggregation not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {AGGREGATIONS}")
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective must be one of {OBJECTIVES}")
        if self.ue_variance < 0 or self.log_base <= 1 or self.episode_length < 1:
            raise ValueError("invalid ue_variance, log_base or episode_length")

    # ------------------------------------------------------------------ sizes
    @property
    def n_aps(self) -> int:
        """Total number of APs ``M + N``."""
        return self.n_tx + self.n_rx

    @property
    def n_users(self) -> int:
        """Number ``K`` of UEs."""
        return len(self.ue_centers)

    @property
    def obs_dim(self) -> int:
        """Dimension of the state of Eq. (18): ``2 K + 2``."""
        return 2 * self.n_users + 2

    @property
    def action_dim(self) -> int:
        """Dimension of the action of Eq. (19): ``2 (M + N)``."""
        return 2 * self.n_aps

    # --------------------------------------------------------------- geometry
    def trajectory(self) -> FloatArray:
        """The ``Q`` sampled points ``p(eps)`` of the circular target trajectory, ``[Q, 2]``."""
        angles = np.linspace(
            0.0, 2.0 * np.pi, self.n_trajectory_points, endpoint=self.trajectory_endpoint
        )
        cx, cy = self.target_center
        return np.stack(
            [cx + self.target_radius * np.cos(angles), cy + self.target_radius * np.sin(angles)],
            axis=1,
        )

    def sample_ues(self, rng: np.random.Generator, n_snapshots: int | None = None) -> FloatArray:
        """Draw UE positions, one snapshot of the ``K`` UEs per trajectory instant.

        Each UE is Gaussian around its centre with per-coordinate variance ``ue_variance``
        (Section 4).  Draws are clipped to the observation bounds.

        Returns:
            Array ``[n_snapshots, K, 2]`` (``n_snapshots`` defaults to ``Q``).
        """
        n = self.n_trajectory_points if n_snapshots is None else n_snapshots
        centers = np.asarray(self.ue_centers, dtype=np.float64)
        noise = rng.normal(0.0, np.sqrt(self.ue_variance), size=(n, self.n_users, 2))
        low, high = self.obs_bounds()
        return np.clip(centers[None] + noise, low[:2], high[:2])

    def obs_bounds(self) -> tuple[FloatArray, FloatArray]:
        """Lower and upper bounds of the observation vector (metres)."""
        low_xy = np.array([self.x_min - self.obs_margin, self.y_min - self.obs_margin])
        high_xy = np.array([self.x_max + self.obs_margin, self.y_max + self.obs_margin])
        reps = self.n_users + 1
        return np.tile(low_xy, reps), np.tile(high_xy, reps)

    def candidate_sites(self) -> FloatArray:
        """All candidate AP sites of the discretised action space, ``[G, 2]``."""
        if self.grid_step is None:
            raise ValueError("candidate sites exist only for a discretised action space")
        xs = np.arange(self.x_min, self.x_max + 1e-9, self.grid_step)
        ys = np.arange(self.y_min, self.y_max + 1e-9, self.grid_step)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        return np.stack([gx.ravel(), gy.ravel()], axis=1)

    def snap(self, positions: ArrayLike) -> FloatArray:
        """Clip positions to the region (Eq. 16) and snap them to the candidate grid."""
        pos = np.asarray(positions, dtype=np.float64).reshape(-1, 2)
        low = np.array([self.x_min, self.y_min])
        high = np.array([self.x_max, self.y_max])
        pos = np.clip(pos, low, high)
        if self.grid_step is not None:
            pos = low + np.round((pos - low) / self.grid_step) * self.grid_step
            pos = np.clip(pos, low, high)
        return pos

    def action_to_positions(self, action: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Map an action in ``[-1, 1]^(2(M+N))`` to transmit and receive AP positions.

        The action is laid out as in Eq. (19): ``(t_1, ..., t_M, r_1, ..., r_N)`` with
        ``(x, y)`` per AP.  ``-1`` maps to the lower and ``+1`` to the upper region bound.
        """
        act = np.clip(np.asarray(action, dtype=np.float64).reshape(self.n_aps, 2), -1.0, 1.0)
        low = np.array([self.x_min, self.y_min])
        high = np.array([self.x_max, self.y_max])
        pos = self.snap(low + 0.5 * (act + 1.0) * (high - low))
        return pos[: self.n_tx], pos[self.n_tx :]

    def positions_to_action(self, tx: ArrayLike, rx: ArrayLike) -> FloatArray:
        """Inverse of :meth:`action_to_positions` (without snapping)."""
        pos = np.concatenate(
            [np.asarray(tx, dtype=np.float64).reshape(-1, 2),
             np.asarray(rx, dtype=np.float64).reshape(-1, 2)]
        )
        low = np.array([self.x_min, self.y_min])
        high = np.array([self.x_max, self.y_max])
        return (2.0 * (pos - low) / (high - low) - 1.0).ravel()

    # -------------------------------------------------------------- objective
    def evaluate(self, tx: ArrayLike, rx: ArrayLike, ues: ArrayLike) -> ObjectiveBreakdown:
        """Factors of Eq. (17) for a deployment and the given UE snapshots."""
        return evaluate_deployment(
            tx,
            rx,
            self.trajectory(),
            ues,
            sensing_metric=self.sensing_metric,
            log_base=self.log_base,
            one_plus_snr=self.one_plus_snr,
            min_distance=self.min_distance,
        )

    def objective_value(self, breakdown: ObjectiveBreakdown) -> float:
        """Scalar objective (Eq. 17 or a Table 3 variant) selected by this configuration."""
        return breakdown.value(self.aggregation, self.objective, self.weight)

    # --------------------------------------------------------------------- io
    def to_dict(self) -> dict[str, Any]:
        """Plain-Python representation (lists instead of tuples), YAML/JSON friendly."""
        out = asdict(self)
        out["target_center"] = list(self.target_center)
        out["ue_centers"] = [list(c) for c in self.ue_centers]
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ScenarioConfig:
        """Build a configuration from a mapping, rejecting unknown keys."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown scenario fields: {sorted(unknown)}")
        return cls(**data)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ScenarioConfig:
        """Load a configuration from a YAML file."""
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle) or {})

    def with_updates(self, **changes: Any) -> ScenarioConfig:
        """Return a copy with some fields replaced."""
        return replace(self, **changes)


def archived_scenario(n_tx: int = 2, n_rx: int = 2, aggregation: str = "max-sum") -> ScenarioConfig:
    """Scenario that reproduces the values in the archived experiment logs.

    It differs from the defaults in two fields only: ``sensing_metric="archived"`` and the
    trajectory centre ``(25.05, 25.05)``.  With these, the archived ``best_rewards_s`` of all
    77 distinct recorded deployments are reproduced to a relative error below ``1e-6``
    (see ``tests/test_archive.py``).
    """
    return ScenarioConfig(
        n_tx=n_tx,
        n_rx=n_rx,
        aggregation=aggregation,
        sensing_metric="archived",
        target_center=(25.05, 25.05),
    )
