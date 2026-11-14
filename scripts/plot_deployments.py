#!/usr/bin/env python
"""Deployment maps in the style of Figs. 4-5 for the optimised deployments of the tables.

Reads ``results/tables/tables_<metric>.json`` (written by ``scripts/make_tables.py``) and
draws the Table 3 deployments: ISAC, sensing-only and communication-only optima of the
max-sum and max-min problems.

    python scripts/plot_deployments.py --tables results/tables --out results/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cfisac.plotting import plot_deployment
from cfisac.scenario import ScenarioConfig

PANELS = {"isac": "ISAC", "sensing": "sensing only", "communication": "communication only"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tables", default="results/tables")
    parser.add_argument("--out", default="results/figures")
    parser.add_argument("--metrics", nargs="+", default=["archived", "eq8"])
    args = parser.parse_args()
    for metric in args.metrics:
        data = json.loads((Path(args.tables) / f"tables_{metric}.json").read_text("utf-8"))
        count = data["table3_n_tx"]
        scenario = ScenarioConfig.from_yaml(f"configs/{metric}/maxsum_{count}t{count}r.yaml")
        for objective, label in PANELS.items():
            for problem in ("max_sum", "max_min"):
                entry = data["table3"][objective][problem]
                tag = problem.replace("_", "")
                name = f"deployment_{metric}_{count}t{count}r_{tag}_{objective}"
                title = f"{problem.replace('_', '-')}, {label} optimum ({metric} sensing value)"
                out = plot_deployment(scenario, entry["tx"], entry["rx"],
                                      Path(args.out) / f"{name}.png", title=title)
                print(f"wrote {out}")


if __name__ == "__main__":
    main()
