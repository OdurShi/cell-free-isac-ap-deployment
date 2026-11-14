"""Scenario data shipped under ``data/``: trajectory samples, candidate sites, UE positions.

The paper uses no external dataset; its training data are produced by the environment.  This
module materialises that data with fixed seeds so that the training and evaluation sets are
fixed files that can be inspected:

* ``ue_positions_train.npy`` -- UE positions used by the training environment,
* ``ue_positions_eval.npy`` -- UE positions used for evaluation and by the optimisers,
* ``trajectory_<family>.csv`` -- the sampled target trajectory of each configuration family,
* ``candidate_sites.csv``, ``ue_centers.csv`` -- the AP grid and the UE cluster centres,
* ``manifest.json`` -- generator parameters, seeds, shapes and SHA-256 of every file.

Loaders return ``None`` when a file is absent or was generated for a different UE
distribution; callers then draw the positions on the fly from the same distribution.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from cfisac.scenario import ScenarioConfig, archived_scenario

FloatArray = NDArray[np.floating]

#: Environment variable that overrides the location of the data directory.
DATA_ENV_VAR = "CFISAC_DATA_DIR"
MANIFEST = "manifest.json"
SPLITS = {
    # split -> (file name, seed, number of UE snapshots, dtype)
    "train": ("ue_positions_train.npy", 1, 409_600, "float32"),
    "eval": ("ue_positions_eval.npy", 0, 32_768, "float64"),
}
#: Seed of the evaluation split; :class:`cfisac.optimize.DeterministicObjective` uses the
#: shipped file when it is asked for this seed.
EVAL_SEED = SPLITS["eval"][1]
TRAJECTORY_FAMILIES = {"eq8": ScenarioConfig(), "archived": archived_scenario()}


def default_data_dir() -> Path | None:
    """Directory holding the shipped data, or ``None`` if there is none.

    ``$CFISAC_DATA_DIR`` takes precedence; otherwise ``data/`` at the root of a source
    checkout (the directory two levels above this package) is used when it exists.
    """
    override = os.environ.get(DATA_ENV_VAR)
    if override:
        return Path(override)
    candidate = Path(__file__).resolve().parents[2] / "data"
    return candidate if (candidate / MANIFEST).is_file() else None


def _ue_parameters(scenario: ScenarioConfig) -> dict[str, Any]:
    """The scenario fields that determine the distribution of the UE positions."""
    low, high = scenario.obs_bounds()
    return {
        "ue_centers": [list(c) for c in scenario.ue_centers],
        "ue_variance": scenario.ue_variance,
        "clip_low": low[:2].tolist(),
        "clip_high": high[:2].tolist(),
    }


def sha256(path: Path) -> str:
    """SHA-256 of a file as a hexadecimal string."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, header: tuple[str, ...], rows: list[tuple]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def generate(out_dir: str | Path, scenario: ScenarioConfig | None = None) -> dict[str, Any]:
    """Generate every data file and the manifest.

    UE positions are drawn with ``ScenarioConfig.sample_ues`` -- Gaussian around the UE
    centres with the configured variance (Section 4 of the paper) -- from
    ``numpy.random.default_rng(seed)`` with the seeds listed in :data:`SPLITS`.

    Args:
        out_dir: target directory (created if needed).
        scenario: scenario whose UE distribution is sampled (defaults to the shared defaults
            of all shipped configurations).

    Returns:
        The manifest that was written.
    """
    cfg = scenario if scenario is not None else ScenarioConfig()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}

    for split, (name, seed, count, dtype) in SPLITS.items():
        positions = cfg.sample_ues(np.random.default_rng(seed), count).astype(dtype)
        np.save(out / name, positions)
        files[name] = {
            "split": split,
            "seed": seed,
            "shape": list(positions.shape),
            "dtype": dtype,
            "snapshots": count,
            "steps_of_25_snapshots": count // 25,
        }

    for family, family_cfg in TRAJECTORY_FAMILIES.items():
        name = f"trajectory_{family}.csv"
        angles = np.linspace(
            0.0, 2.0 * np.pi, family_cfg.n_trajectory_points,
            endpoint=family_cfg.trajectory_endpoint,
        )
        rows = [
            (i, repr(float(a)), repr(float(x)), repr(float(y)))
            for i, (a, (x, y)) in enumerate(zip(angles, family_cfg.trajectory(), strict=True))
        ]
        _write_csv(out / name, ("index", "angle_rad", "x_m", "y_m"), rows)
        files[name] = {
            "points": len(rows),
            "center": list(family_cfg.target_center),
            "radius": family_cfg.target_radius,
        }

    sites = cfg.candidate_sites()
    _write_csv(out / "candidate_sites.csv", ("index", "x_m", "y_m"),
               [(i, repr(float(x)), repr(float(y))) for i, (x, y) in enumerate(sites)])
    files["candidate_sites.csv"] = {"sites": len(sites), "grid_step": cfg.grid_step}
    _write_csv(out / "ue_centers.csv", ("ue", "x_m", "y_m"),
               [(k + 1, repr(x), repr(y)) for k, (x, y) in enumerate(cfg.ue_centers)])
    files["ue_centers.csv"] = {"users": cfg.n_users}

    for name, entry in files.items():
        entry["bytes"] = (out / name).stat().st_size
        entry["sha256"] = sha256(out / name)
    manifest = {
        "generator": "cfisac.data.generate (scripts/generate_data.py)",
        "numpy_version": np.__version__,
        "ue_distribution": _ue_parameters(cfg),
        "files": files,
    }
    with open(out / MANIFEST, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    return manifest


def load_manifest(data_dir: str | Path | None = None) -> dict[str, Any] | None:
    """Manifest of a data directory, or ``None`` if the directory has none."""
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    if root is None or not (root / MANIFEST).is_file():
        return None
    with open(root / MANIFEST, encoding="utf-8") as handle:
        return json.load(handle)


def load_ue_positions(
    split: str, scenario: ScenarioConfig, data_dir: str | Path | None = None
) -> FloatArray | None:
    """UE positions of a split, shape ``[snapshots, K, 2]``, if they fit the scenario.

    Returns ``None`` when no data directory is available, the file is missing, or the data
    were generated for another UE distribution (centres, variance or clipping bounds).
    """
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {tuple(SPLITS)}")
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    manifest = load_manifest(root)
    if root is None or manifest is None:
        return None
    path = root / SPLITS[split][0]
    if not path.is_file() or manifest.get("ue_distribution") != _ue_parameters(scenario):
        return None
    return np.load(path, allow_pickle=False)


def load_trajectory(family: str, data_dir: str | Path | None = None) -> FloatArray | None:
    """Trajectory samples ``[Q, 2]`` of a configuration family from its CSV file."""
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    if root is None or not (root / f"trajectory_{family}.csv").is_file():
        return None
    with open(root / f"trajectory_{family}.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return np.array([[float(r["x_m"]), float(r["y_m"])] for r in rows], dtype=np.float64)


def verify(data_dir: str | Path | None = None) -> dict[str, bool]:
    """Compare every shipped file with the SHA-256 recorded in the manifest."""
    root = Path(data_dir) if data_dir is not None else default_data_dir()
    manifest = load_manifest(root)
    if root is None or manifest is None:
        raise FileNotFoundError("no data directory with a manifest was found")
    return {
        name: (root / name).is_file() and sha256(root / name) == entry["sha256"]
        for name, entry in manifest["files"].items()
    }
