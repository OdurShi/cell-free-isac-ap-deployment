"""Slow, loop-based reference implementations of the paper's formulas.

They use only the ``math`` module and follow the printed equations term by term.  The unit
tests compare :mod:`cfisac.metrics` against them; nothing else should import this module.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = Sequence[float]


def _direction(ap: Point, target: Point) -> tuple[float, float]:
    dx, dy = target[0] - ap[0], target[1] - ap[1]
    dist = math.hypot(dx, dy)
    return dx / dist, dy / dist


def fim_sums_reference(
    tx: Sequence[Point], rx: Sequence[Point], target: Point
) -> tuple[float, float, float]:
    """The three double sums of Eq. (8) for one target position."""
    a = b = c = 0.0
    for t in tx:
        ctx, stx = _direction(t, target)
        for r in rx:
            crx, srx = _direction(r, target)
            a += (ctx + crx) ** 2
            b += (stx + srx) ** 2
            c += (ctx + crx) * (stx + srx)
    return a, b, c


def fim_determinant_reference(tx: Sequence[Point], rx: Sequence[Point], target: Point) -> float:
    """Eq. (8): ``A * B - C**2`` for one target position."""
    a, b, c = fim_sums_reference(tx, rx, target)
    return a * b - c * c


def fim_determinant_lagrange(tx: Sequence[Point], rx: Sequence[Point], target: Point) -> float:
    """Eq. (8) through Lagrange's identity.

    With ``v_mn = alpha^t_m + alpha^r_n`` the rows of ``J0`` (Eq. 6), the determinant of
    ``J0^T J0`` equals the sum of squared 2-D cross products over all unordered row pairs.
    This is an independent route to the same number and shows that it is non-negative.
    """
    rows = []
    for t in tx:
        ctx, stx = _direction(t, target)
        for r in rx:
            crx, srx = _direction(r, target)
            rows.append((ctx + crx, stx + srx))
    total = 0.0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            total += (rows[i][0] * rows[j][1] - rows[i][1] * rows[j][0]) ** 2
    return total


def snr_reference(aps: Sequence[Point], ue: Point) -> float:
    """Eq. (14) for one UE."""
    return sum(1.0 / ((ue[0] - a[0]) ** 2 + (ue[1] - a[1]) ** 2) for a in aps)


def rate_reference(aps: Sequence[Point], ue: Point, log_base: float = 10.0) -> float:
    """Eq. (15), first line: ``log(1 + SNR_k)``."""
    return math.log(1.0 + snr_reference(aps, ue), log_base)


def objective_reference(
    tx: Sequence[Point],
    rx: Sequence[Point],
    targets: Sequence[Point],
    ues: Sequence[Sequence[Point]],
    aggregation: str,
    log_base: float = 10.0,
) -> float:
    """Eq. (17) for UE snapshots ``ues[q][k]`` (one snapshot per time instant)."""
    aps = list(tx) + list(rx)
    dets = [fim_determinant_reference(tx, rx, p) for p in targets]
    n_users = len(ues[0])
    per_user = [
        sum(rate_reference(aps, snapshot[k], log_base) for snapshot in ues) / len(ues)
        for k in range(n_users)
    ]
    if aggregation == "max-sum":
        return sum(per_user) * (sum(dets) / len(dets))
    if aggregation == "max-min":
        return min(per_user) * min(dets)
    raise ValueError(aggregation)
