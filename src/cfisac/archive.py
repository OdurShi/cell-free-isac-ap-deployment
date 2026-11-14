"""Consistency checks between this implementation and the archived experiment logs.

The archived ``experiments/`` tree stores, for every evaluation of the original study, the
best deployment so far (``recorder_best_location.csv``) and its per-step sensing and
communication factors (``best_rewards_s`` and ``rewards_c / 256`` in ``recorder_my.csv``).
These functions recompute the factors with this code base so that every scenario constant
calibrated against the archive can be re-derived (``scripts/analyze_archive.py``).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from cfisac.metrics import fim_determinant
from cfisac.optimize import DeterministicObjective
from cfisac.scenario import ScenarioConfig

FloatArray = NDArray[np.float64]

#: Steps per evaluation episode in the archived recorders (cumulative columns / 256).
ARCHIVE_EPISODE_LENGTH = 256


@dataclass(frozen=True)
class RecordedDeployment:
    """One distinct deployment found in the archive.

    Attributes:
        tx, rx: AP positions (first half / second half of a best-location row).
        sens: recorded per-step sensing factor (``best_rewards_s``).
        comm: recorded mean per-step communication factor (``rewards_c / 256``), averaged
            over the evaluation rows in which the evaluated policy played this deployment.
        n_rows: number of such rows.
    """

    tx: FloatArray
    rx: FloatArray
    sens: float
    comm: float
    n_rows: int


def collect_recorded_deployments(root: str | Path) -> list[RecordedDeployment]:
    """Distinct deployments of the archive with their recorded factors.

    A row contributes to ``comm`` only if the evaluated policy's own sensing value
    (``rewards_s / 256``) equals the best deployment's (``best_rewards_s``), i.e. when the
    evaluated deployment is the recorded one.
    """
    sens: dict[str, float] = {}
    comm: dict[str, list[float]] = {}
    for recorder in sorted(Path(root).glob("*/*/seed-*/recorder_my.csv")):
        with open(recorder, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        locations = (recorder.parent / "recorder_best_location.csv").read_text("utf-8").split()
        for row, location in zip(rows, locations, strict=True):
            best_s = float(row["best_rewards_s"])
            sens[location] = best_s
            if abs(float(row["rewards_s"]) / ARCHIVE_EPISODE_LENGTH - best_s) <= 1e-4 * best_s:
                comm.setdefault(location, []).append(
                    float(row["rewards_c"]) / ARCHIVE_EPISODE_LENGTH
                )
    out = []
    for location in sorted(sens):
        aps = np.array([float(v) for v in location.split(",")]).reshape(-1, 2)
        half = len(aps) // 2
        values = comm.get(location, [])
        out.append(
            RecordedDeployment(
                tx=aps[:half],
                rx=aps[half:],
                sens=sens[location],
                comm=float(np.mean(values)) if values else float("nan"),
                n_rows=len(values),
            )
        )
    return out


def sensing_relative_errors(
    records: list[RecordedDeployment], scenario: ScenarioConfig
) -> FloatArray:
    """Relative deviation of the recomputed trajectory-mean sensing value from the archive."""
    trajectory = scenario.trajectory()
    errors = []
    for rec in records:
        value = fim_determinant(
            rec.tx, rec.rx, trajectory, scenario.sensing_metric, scenario.min_distance
        ).mean()
        errors.append(abs(value - rec.sens) / abs(rec.sens))
    return np.array(errors)


def rate_relative_errors(
    records: list[RecordedDeployment],
    scenario: ScenarioConfig,
    n_ue_snapshots: int = 4096,
    seed: int = 0,
) -> FloatArray:
    """Relative deviation of the recomputed mean sum rate from the archive (signed)."""
    errors = []
    for rec in records:
        if rec.n_rows == 0:
            continue
        cfg = scenario.with_updates(n_tx=len(rec.tx), n_rx=len(rec.rx))
        value = DeterministicObjective(cfg, n_ue_snapshots, seed).breakdown(rec.tx, rec.rx).comm_sum
        errors.append((value - rec.comm) / rec.comm)
    return np.array(errors)
