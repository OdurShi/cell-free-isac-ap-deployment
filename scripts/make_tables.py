#!/usr/bin/env python
"""Tables 2 and 3 from deployments found by the non-RL optimisers, for both sensing values.

    python scripts/make_tables.py --out results/tables

Writes ``tables_archived.{json,md}`` and ``tables_eq8.{json,md}`` plus
``figure_deployments.md`` (the deployments shown in Figs. 4-5 evaluated in the archived
4T4R scenario, next to the paper's Table 3).
"""

from __future__ import annotations

import argparse

from cfisac.cli import main as cfisac_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="results/tables")
    parser.add_argument("--ue-snapshots", default="1024", help="UE snapshots per score")
    parser.add_argument("--n-starts", default="8", help="random starts of the local search")
    parser.add_argument("--counts", nargs="+", default=["1", "2", "3", "4"])
    args = parser.parse_args()
    for metric in ("archived", "eq8"):
        scenario = f"configs/{metric}/maxsum_2t2r.yaml"
        cfisac_main(["tables", "--scenario", scenario, "--counts", *args.counts,
                     "--table3-count", "4", "--ue-snapshots", args.ue_snapshots,
                     "--n-starts", args.n_starts, "--out", args.out])
    cfisac_main(["check-table3", "--out", f"{args.out}/figure_deployments.md"])


if __name__ == "__main__":
    main()
