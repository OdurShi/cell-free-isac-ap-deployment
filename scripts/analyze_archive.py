#!/usr/bin/env python
"""Re-derive the scenario constants that were calibrated against the archived logs.

For every distinct deployment recorded under ``experiments/`` the script recomputes the
sensing factor for several candidate settings and the communication factor for three
logarithm bases, and reports how far each candidate is from the recorded values.

    python scripts/analyze_archive.py --archive experiments --out results/archive_analysis.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cfisac.archive import (
    collect_recorded_deployments,
    rate_relative_errors,
    sensing_relative_errors,
)
from cfisac.scenario import ScenarioConfig, archived_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--archive", default="experiments")
    parser.add_argument("--out", default="results/archive_analysis.md")
    parser.add_argument("--ue-snapshots", type=int, default=4096)
    args = parser.parse_args()

    records = collect_recorded_deployments(args.archive)
    n_1t1r = sum(len(r.tx) == 1 for r in records)
    lines = [
        "# Archive consistency analysis",
        "",
        f"Distinct recorded deployments: {len(records)} ({n_1t1r} with 1T1R, "
        f"{len(records) - n_1t1r} with 2T2R).",
        "",
        "## Sensing factor (`best_rewards_s`) under candidate settings",
        "",
        "| sensing value | trajectory centre | trajectory samples | max rel. error | median |",
        "|---|---|---|---|---|",
    ]
    base = archived_scenario()
    candidates = [
        ("A*B - C (`archived`)", base, "(25.05, 25.05)", "linspace(0, 2pi, 25), endpoint"),
        ("A*B - C (`archived`)", base.with_updates(target_center=(25.0, 25.0)),
         "(25, 25)", "linspace(0, 2pi, 25), endpoint"),
        ("A*B - C (`archived`)",
         base.with_updates(n_trajectory_points=24, trajectory_endpoint=False),
         "(25.05, 25.05)", "24 distinct points"),
        ("A*B - C (`archived`)",
         base.with_updates(n_trajectory_points=256, trajectory_endpoint=False),
         "(25.05, 25.05)", "256 distinct points"),
        ("A*B - C^2 (`eq8`)", base.with_updates(sensing_metric="eq8"),
         "(25.05, 25.05)", "linspace(0, 2pi, 25), endpoint"),
    ]
    for name, scenario, centre, samples in candidates:
        err = sensing_relative_errors(records, scenario)
        lines.append(f"| {name} | {centre} | {samples} | {err.max():.2e} | {np.median(err):.2e} |")

    lines += [
        "",
        "## Communication factor (`rewards_c / 256`) under candidate logarithm bases",
        "",
        "Recomputed as the mean sum rate over a fixed sample of UE snapshots (UE centres "
        f"{list(map(list, ScenarioConfig().ue_centers))}, variance 2).",
        "",
        "| log base | median signed rel. error | median abs. rel. error | max abs. rel. error |",
        "|---|---|---|---|",
    ]
    for label, log_base in (("10", 10.0), ("e", float(np.e)), ("2", 2.0)):
        err = rate_relative_errors(records, base.with_updates(log_base=log_base),
                                   args.ue_snapshots)
        lines.append(
            f"| {label} | {np.median(err):+.3f} | {np.median(np.abs(err)):.3f} "
            f"| {np.abs(err).max():.3f} |"
        )
    text = "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
