#!/usr/bin/env python
"""Crop the numbered figures of the paper out of its PDF.

    python scripts/extract_paper_figures.py /path/to/paper.pdf --out assets/paper_figures

Needs PyMuPDF (``pip install pymupdf``).  The PDF is only read.  Each figure is rendered from
the crop box listed in ``FIGURES`` (PDF points, origin at the top-left corner of the page);
the boxes enclose the graphics and the sub-panel labels, not the caption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# name -> (1-based page number, (x0, y0, x1, y1) in PDF points)
FIGURES: dict[str, tuple[int, tuple[float, float, float, float]]] = {
    "fig01": (2, (334.0, 81.0, 542.0, 251.0)),
    "fig02": (4, (316.0, 165.0, 560.0, 292.0)),
    "fig03": (5, (322.0, 86.0, 554.0, 440.0)),
    "fig04": (6, (55.0, 86.0, 293.0, 430.0)),
    "fig05": (6, (319.0, 86.0, 557.0, 441.0)),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pdf", help="path to the paper PDF (read-only)")
    parser.add_argument("--out", default="assets/paper_figures", help="output directory")
    parser.add_argument("--dpi", type=int, default=300, help="rendering resolution")
    args = parser.parse_args()

    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - depends on the optional package
        raise SystemExit("PyMuPDF is required: pip install pymupdf") from exc

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with fitz.open(args.pdf) as document:
        for name, (page_number, box) in FIGURES.items():
            page = document[page_number - 1]
            pixmap = page.get_pixmap(clip=fitz.Rect(*box), dpi=args.dpi)
            target = out / f"{name}.png"
            pixmap.save(target)
            print(f"{target}  {pixmap.width}x{pixmap.height} px  page {page_number}")


if __name__ == "__main__":
    main()
