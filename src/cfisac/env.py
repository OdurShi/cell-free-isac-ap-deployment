"""Gymnasium environment for the MDP of Section 3.1."""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from cfisac.scenario import ScenarioConfig


class APDeploymentEnv(gym.Env):
    """AP deployment for cell-free ISAC as a Markov decision process.

    * **State** (Eq. 18): the 2-D coordinates of the ``K`` UEs and of the target,
      ``s_i = {u_1, ..., u_K, p}``, a vector of length ``2 K + 2`` in metres.
    * **Action** (Eq. 19): the 2-D coordinates of the ``M`` transmit and ``N`` receive APs,
      a vector in ``[-1, 1]^(2(M+N))`` that is mapped affinely onto the deployment region
      of Eq. (16) and snapped to the candidate grid (the paper discretises the action space).
    * **Reward** (Eqs. 17/20): the integrated objective of the chosen deployment, evaluated
      over the whole sampled trajectory and one UE snapshot per trajectory instant.

    Dynamics: at every step UE positions for the ``Q`` trajectory instants are taken -- a
    random block of ``Q`` snapshots of ``ue_positions`` when such a pool is given (the
    shipped ``data/`` files, see :mod:`cfisac.data`), otherwise fresh Gaussian draws.  The
    observation shows the snapshot of the current instant (target point ``i mod Q`` and the
    UE positions belonging to it).  The transition does not depend on the action -- the
    paper's formulation is a deployment (bandit-like) problem cast as an MDP.  Episodes are
    truncated after ``episode_length`` steps and never terminate.

    ``info`` contains the Table 3 quantities (``comm_sum``, ``comm_min``, ``sens_mean``,
    ``sens_min``), the selected factors ``comm`` and ``sens``, their product ``isac`` and the
    snapped AP positions ``tx`` and ``rx``.
    """

    metadata = {"render_modes": []}

    def __init__(
        self, config: ScenarioConfig | None = None, ue_positions: np.ndarray | None = None
    ) -> None:
        """Create the environment.

        Args:
            config: scenario; defaults to :class:`ScenarioConfig`.
            ue_positions: optional pool of UE snapshots, shape ``[S, K, 2]`` with
                ``S >= Q``.  If given, every step uses ``Q`` consecutive snapshots of the
                pool (a block chosen uniformly at random); if ``None``, positions are drawn
                from the scenario's Gaussian UE distribution at every step.
        """
        super().__init__()
        self.config = config if config is not None else ScenarioConfig()
        self._pool: np.ndarray | None = None
        self._n_blocks = 0
        if ue_positions is not None:
            pool = np.asarray(ue_positions)
            q = self.config.n_trajectory_points
            if pool.ndim != 3 or pool.shape[1:] != (self.config.n_users, 2) or len(pool) < q:
                raise ValueError(
                    f"ue_positions must have shape [S >= {q}, {self.config.n_users}, 2], "
                    f"got {pool.shape}"
                )
            self._pool = pool
            self._n_blocks = len(pool) // q
        low, high = self.config.obs_bounds()
        self.observation_space = spaces.Box(
            low=low.astype(np.float32), high=high.astype(np.float32), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.config.action_dim,), dtype=np.float32
        )
        self._trajectory = self.config.trajectory()
        self._ues = np.zeros((self.config.n_trajectory_points, self.config.n_users, 2))
        self._t = 0

    def _draw_snapshot(self) -> np.ndarray:
        """Take the UE positions of this step and build the observation of Eq. (18)."""
        q = self.config.n_trajectory_points
        if self._pool is None:
            self._ues = self.config.sample_ues(self.np_random)
        else:
            block = int(self.np_random.integers(self._n_blocks))
            self._ues = np.asarray(self._pool[block * q : (block + 1) * q], dtype=np.float64)
        idx = self._t % q
        obs = np.concatenate([self._ues[idx].ravel(), self._trajectory[idx]])
        return obs.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._t = 0
        return self._draw_snapshot(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        cfg = self.config
        tx, rx = cfg.action_to_positions(action)
        breakdown = cfg.evaluate(tx, rx, self._ues)
        reward = cfg.reward_scale * cfg.objective_value(breakdown)
        info = {
            "comm_sum": breakdown.comm_sum,
            "comm_min": breakdown.comm_min,
            "sens_mean": breakdown.sens_mean,
            "sens_min": breakdown.sens_min,
            "comm": breakdown.communication(cfg.aggregation),
            "sens": breakdown.sensing(cfg.aggregation),
            "isac": breakdown.isac(cfg.aggregation),
            "tx": tx,
            "rx": rx,
        }
        self._t += 1
        truncated = self._t >= cfg.episode_length
        return self._draw_snapshot(), float(reward), False, truncated, info
