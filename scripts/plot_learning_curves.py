#!/usr/bin/env python
"""Learning curves in the style of Fig. 3.

From the archived recorders of the original study (default) or from new runs:

    python scripts/plot_learning_curves.py
    python scripts/plot_learning_curves.py --runs results/runs/paper_hparams_maxsum \\
        --out results/figures/short_runs_maxsum.png --title "This repository, 2T2R max-sum"
"""

from __future__ import annotations

import argparse

from cfisac.cli import main as cfisac_main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", default="experiments")
    parser.add_argument("--setting", default="2t2r")
    parser.add_argument("--max-steps", default="3e6")
    parser.add_argument("--title", default="Archived runs of the original study: 2T2R, max-sum")
    parser.add_argument("--out", default="results/figures/fig3a_archived_2t2r_maxsum.png")
    args = parser.parse_args()
    raise SystemExit(
        cfisac_main(["curves", "--runs", args.runs, "--setting", args.setting,
                     "--max-steps", args.max_steps, "--title", args.title, "--out", args.out])
    )


if __name__ == "__main__":
    main()
