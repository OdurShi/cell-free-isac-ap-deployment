#!/usr/bin/env python
"""Generate the scenario data shipped under ``data/``.

    python scripts/generate_data.py --out data

Writes the UE positions of the training and evaluation splits, the sampled target
trajectories, the candidate AP sites, the UE centres and ``manifest.json`` (seeds, shapes,
SHA-256).  The output is determined by the seeds in ``cfisac.data.SPLITS``.
"""

from __future__ import annotations

import argparse

from cfisac.data import generate, verify


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="data", help="output directory")
    args = parser.parse_args()
    manifest = generate(args.out)
    for name, entry in manifest["files"].items():
        print(f"{name:28s} {entry['bytes']:>10d} bytes  sha256 {entry['sha256']}")
    assert all(verify(args.out).values())


if __name__ == "__main__":
    main()
