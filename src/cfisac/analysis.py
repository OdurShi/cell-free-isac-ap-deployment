"""Reading training recorders (archived or new) and aggregating them across seeds."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: File names of a recorder: the archived study wrote ``recorder_my.csv``; this code writes
#: ``recorder.csv`` with the same columns.
RECORDER_NAMES = ("recorder.csv", "recorder_my.csv")


@dataclass(frozen=True)
class Run:
    """One training run found on disk."""

    algo: str
    setting: str
    seed: int
    path: Path


def load_recorder(path: str | Path) -> dict[str, FloatArray]:
    """Load a recorder CSV into a mapping ``column -> array``.

    The unnamed first column (a row index) is returned under the key ``"index"``.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"empty recorder: {path}")
    header = [name if name else "index" for name in rows[0]]
    data = np.array([[float(v) for v in row] for row in rows[1:] if row], dtype=np.float64)
    if data.size == 0:
        data = np.zeros((0, len(header)))
    return {name: data[:, i] for i, name in enumerate(header)}


def load_best_locations(path: str | Path) -> FloatArray:
    """Load ``recorder_best_location.csv``: one deployment per row, shape ``[rows, L, 2]``."""
    with open(path, newline="", encoding="utf-8") as handle:
        rows = [[float(v) for v in row] for row in csv.reader(handle) if row]
    return np.array(rows, dtype=np.float64).reshape(len(rows), -1, 2)


def discover_runs(root: str | Path) -> list[Run]:
    """Find runs laid out as ``<root>/<algo>/<setting>/seed-<n>/recorder*.csv``.

    This is the layout of the archived ``experiments/`` tree and of :func:`cfisac.training.train`
    when used through the CLI.  Runs are returned sorted by (algo, setting, seed).
    """
    runs = []
    for name in RECORDER_NAMES:
        for file in Path(root).glob(f"*/*/seed-*/{name}"):
            seed_dir = file.parent
            runs.append(
                Run(
                    algo=seed_dir.parent.parent.name,
                    setting=seed_dir.parent.name,
                    seed=int(seed_dir.name.split("-", 1)[1]),
                    path=file,
                )
            )
    return sorted(runs, key=lambda r: (r.algo, r.setting, r.seed))


@dataclass(frozen=True)
class Curve:
    """Mean and spread of one recorder column across the seeds of an algorithm."""

    steps: FloatArray
    mean: FloatArray
    std: FloatArray
    low: FloatArray
    high: FloatArray
    n_seeds: int


def aggregate_runs(
    runs: list[Run], column: str = "rewards", max_steps: float | None = None, n_points: int = 300
) -> Curve:
    """Aggregate one column over several seeds on a common step grid.

    Each run is linearly interpolated onto ``n_points`` steps between the first evaluation
    and the shortest run's last evaluation (optionally capped at ``max_steps``).
    """
    if not runs:
        raise ValueError("no runs to aggregate")
    series = []
    for run in runs:
        rec = load_recorder(run.path)
        series.append((rec["steps"], rec[column]))
    start = max(s[0] for s, _ in series)
    stop = min(s[-1] for s, _ in series)
    if max_steps is not None:
        stop = min(stop, max_steps)
    grid = np.linspace(start, stop, n_points)
    values = np.stack([np.interp(grid, s, v) for s, v in series])
    return Curve(
        steps=grid,
        mean=values.mean(axis=0),
        std=values.std(axis=0),
        low=values.min(axis=0),
        high=values.max(axis=0),
        n_seeds=len(series),
    )


def final_performance(runs: list[Run], column: str = "rewards", last_fraction: float = 0.1) -> dict:
    """Mean of the last ``last_fraction`` of each run's evaluations, summarised over seeds."""
    finals = []
    for run in runs:
        values = load_recorder(run.path)[column]
        tail = max(1, int(np.ceil(len(values) * last_fraction)))
        finals.append(float(values[-tail:].mean()))
    arr = np.array(finals)
    return {
        "per_seed": {run.seed: value for run, value in zip(runs, finals, strict=True)},
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
