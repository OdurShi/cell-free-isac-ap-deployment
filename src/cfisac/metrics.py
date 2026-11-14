"""Sensing and communication metrics of the paper (Section 2).

All functions are vectorised NumPy. Shapes use the paper's symbols:

* ``M`` transmit APs ``t_m``, ``N`` receive APs ``r_n`` (single antenna each),
* ``K`` user equipments ``u_k``,
* ``Q`` sampled points ``p(eps)`` of the target trajectory.

Equation numbers refer to Xu et al., "Access Point Deployment for Localizing Accuracy and
User Rate in Cell-Free Systems", ACM MobiCom '24.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: ``"eq8"`` is the determinant exactly as printed in Eq. (8).  ``"archived"`` is the quantity
#: found in the archived logs of the original study (cross term not squared); see
#: ``docs/implementation_notes.md``.
SENSING_METRICS = ("eq8", "archived")
AGGREGATIONS = ("max-sum", "max-min")
OBJECTIVES = ("isac", "sensing", "communication", "weighted")

DEFAULT_MIN_DISTANCE = 1e-9


def _as_points(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64).reshape(-1, 2)


def unit_directions(
    aps: ArrayLike, targets: ArrayLike, min_distance: float = DEFAULT_MIN_DISTANCE
) -> FloatArray:
    """Unit vectors pointing from every AP to every target point.

    These are the fractions ``(x_p - x_ap) / ||p - ap||_2`` and ``(y_p - y_ap) / ||p - ap||_2``
    that appear in Eq. (8), i.e. ``(cos(theta), sin(theta))`` of the bearing angle of Eq. (6).

    Args:
        aps: AP positions, shape ``[L, 2]``.
        targets: target positions, shape ``[Q, 2]``.
        min_distance: lower clip of the distance, guards an AP placed exactly on a target point.

    Returns:
        Array of shape ``[Q, L, 2]``.
    """
    diff = _as_points(targets)[:, None, :] - _as_points(aps)[None, :, :]
    dist = np.linalg.norm(diff, axis=-1, keepdims=True)
    return diff / np.maximum(dist, min_distance)


def bearing_angles(aps: ArrayLike, targets: ArrayLike) -> FloatArray:
    """Bearing angles ``theta`` used in Eqs. (6)-(7), shape ``[Q, L]``.

    ``cos(theta) = (x_p - x_ap) / ||p - ap||`` and ``sin(theta) = (y_p - y_ap) / ||p - ap||``,
    which is the convention that turns Eq. (7) into Eq. (8).
    """
    diff = _as_points(targets)[:, None, :] - _as_points(aps)[None, :, :]
    return np.arctan2(diff[..., 1], diff[..., 0])


def fim_terms(
    tx: ArrayLike,
    rx: ArrayLike,
    targets: ArrayLike,
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The three double sums of Eq. (8), one value per trajectory point.

    With ``c_mn = cos(theta^t_m) + cos(theta^r_n)`` and ``s_mn = sin(theta^t_m) + sin(theta^r_n)``:

    ``A = sum_m sum_n c_mn^2``,  ``B = sum_m sum_n s_mn^2``,  ``C = sum_m sum_n c_mn s_mn``.

    ``[[A, C], [C, B]]`` is the non-coherent Fisher information matrix ``J0^T J0`` of Eq. (5)
    with ``Sigma = I``.

    Returns:
        ``(A, B, C)``, each of shape ``[Q]``.
    """
    ut = unit_directions(tx, targets, min_distance)  # [Q, M, 2]
    ur = unit_directions(rx, targets, min_distance)  # [Q, N, 2]
    pair = ut[:, :, None, :] + ur[:, None, :, :]  # [Q, M, N, 2]
    c, s = pair[..., 0], pair[..., 1]
    return (c * c).sum(axis=(1, 2)), (s * s).sum(axis=(1, 2)), (c * s).sum(axis=(1, 2))


def combine_fim_terms(a: FloatArray, b: FloatArray, c: FloatArray, metric: str) -> FloatArray:
    """Turn the sums ``(A, B, C)`` of :func:`fim_terms` into the sensing value.

    * ``"eq8"``: ``A * B - C**2`` -- the determinant ``|Phi|`` of Eqs. (7)-(8).
    * ``"archived"``: ``A * B - C`` -- *not* a determinant.  It is what the archived
      experiment logs of the original study contain (verified numerically, see
      ``docs/implementation_notes.md``) and is offered only to reproduce those logs.
    """
    if metric == "eq8":
        return a * b - c * c
    if metric == "archived":
        return a * b - c
    raise ValueError(f"unknown sensing metric {metric!r}; expected one of {SENSING_METRICS}")


def fim_determinant(
    tx: ArrayLike,
    rx: ArrayLike,
    targets: ArrayLike,
    metric: str = "eq8",
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> FloatArray:
    """Determinant of the non-coherent sensing FIM in Cartesian form, Eq. (8).

    Args:
        tx: transmit AP positions ``[M, 2]``.
        rx: receive AP positions ``[N, 2]``.
        targets: target positions ``[Q, 2]`` (a single point may be passed as ``[2]``).
        metric: ``"eq8"`` (paper) or ``"archived"`` (see :func:`combine_fim_terms`).
        min_distance: lower clip of AP-target distances.

    Returns:
        ``|Phi|`` at every target point, shape ``[Q]``.
    """
    return combine_fim_terms(*fim_terms(tx, rx, targets, min_distance), metric)


def fim_determinant_from_angles(theta_t: ArrayLike, theta_r: ArrayLike) -> float:
    """Determinant of the angular FIM, Eq. (7), for one target position.

    Args:
        theta_t: bearing angles of the ``M`` transmit APs.
        theta_r: bearing angles of the ``N`` receive APs.
    """
    tt = np.asarray(theta_t, dtype=np.float64).reshape(-1, 1)
    tr = np.asarray(theta_r, dtype=np.float64).reshape(1, -1)
    c = np.cos(tt) + np.cos(tr)
    s = np.sin(tt) + np.sin(tr)
    return float((c * c).sum() * (s * s).sum() - (c * s).sum() ** 2)


def snr(
    aps: ArrayLike,
    ues: ArrayLike,
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> FloatArray:
    """Euclidean-distance SNR of every UE, Eq. (14).

    ``SNR_k = sum_m 1 / ||u_k - t_m||^2 + sum_n 1 / ||u_k - r_n||^2`` (unit transmit power and
    reference distance ``d0 = 1`` as assumed below Eq. (13)).  Transmit and receive APs enter
    identically, so they are passed as one array.

    Args:
        aps: all ``M + N`` AP positions, shape ``[L, 2]``.
        ues: UE positions, shape ``[..., K, 2]``.
        min_distance: lower clip of AP-UE distances (keeps the SNR finite).

    Returns:
        SNR per UE, shape ``[..., K]``.
    """
    ue_arr = np.asarray(ues, dtype=np.float64)
    diff = ue_arr[..., None, :] - _as_points(aps)  # [..., K, L, 2]
    dist_sq = np.maximum((diff * diff).sum(axis=-1), min_distance**2)
    return (1.0 / dist_sq).sum(axis=-1)


def user_rates(
    aps: ArrayLike,
    ues: ArrayLike,
    log_base: float = 10.0,
    one_plus_snr: bool = True,
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> FloatArray:
    """Per-UE rate ``R_k``, Eq. (15).

    The first line of Eq. (15) is ``log(1 + SNR_k)``; its last line is printed without the
    ``1 +``.  ``one_plus_snr=True`` (default) follows the first line, which is the one
    consistent with Eq. (14) and with the archived logs; ``False`` evaluates the last line
    literally.  The paper does not state the base of the logarithm; ``log_base`` defaults to
    10, the base that reproduces the archived logs.

    Returns:
        Rates with the shape of ``ues`` minus its last axis, ``[..., K]``.
    """
    value = snr(aps, ues, min_distance)
    arg = 1.0 + value if one_plus_snr else value
    return np.log(arg) / np.log(log_base)


@dataclass(frozen=True)
class ObjectiveBreakdown:
    """The four quantities reported in Table 3 for one deployment.

    Attributes:
        comm_sum: ``sum_k R_k / Q`` -- sum rate averaged over the ``Q`` trajectory instants.
        comm_min: ``min_k R_k`` -- rate of the worst UE (each UE averaged over the instants).
        sens_mean: ``sum_p |Phi| / Q`` -- sensing value averaged over the trajectory.
        sens_min: ``min_p |Phi|`` -- worst sensing value along the trajectory.
    """

    comm_sum: float
    comm_min: float
    sens_mean: float
    sens_min: float

    def communication(self, aggregation: str) -> float:
        """Communication factor of Eq. (17) for the given problem."""
        _check(aggregation, AGGREGATIONS, "aggregation")
        return self.comm_sum if aggregation == "max-sum" else self.comm_min

    def sensing(self, aggregation: str) -> float:
        """Sensing factor of Eq. (17) for the given problem."""
        _check(aggregation, AGGREGATIONS, "aggregation")
        return self.sens_mean if aggregation == "max-sum" else self.sens_min

    def isac(self, aggregation: str) -> float:
        """Integrated objective ``U(t, r)`` of Eq. (17) (product of the two factors)."""
        return self.communication(aggregation) * self.sensing(aggregation)

    def value(self, aggregation: str, objective: str = "isac", weight: float = 0.5) -> float:
        """Objective used for optimisation.

        ``"isac"`` is Eq. (17).  The other three are the comparison objectives of Table 3:
        ``"sensing"`` and ``"communication"`` keep a single factor, ``"weighted"`` is
        ``weight * communication + (1 - weight) * sensing``.
        """
        _check(objective, OBJECTIVES, "objective")
        comm, sens = self.communication(aggregation), self.sensing(aggregation)
        if objective == "isac":
            return comm * sens
        if objective == "sensing":
            return sens
        if objective == "communication":
            return comm
        return weight * comm + (1.0 - weight) * sens


def _check(value: str, allowed: tuple[str, ...], name: str) -> None:
    if value not in allowed:
        raise ValueError(f"unknown {name} {value!r}; expected one of {allowed}")


def evaluate_deployment(
    tx: ArrayLike,
    rx: ArrayLike,
    targets: ArrayLike,
    ues: ArrayLike,
    sensing_metric: str = "eq8",
    log_base: float = 10.0,
    one_plus_snr: bool = True,
    min_distance: float = DEFAULT_MIN_DISTANCE,
) -> ObjectiveBreakdown:
    """Evaluate the factors of Eq. (17) for one deployment.

    Args:
        tx: transmit AP positions ``[M, 2]``.
        rx: receive AP positions ``[N, 2]``.
        targets: the ``Q`` sampled trajectory points ``[Q, 2]``.
        ues: UE positions, either ``[K, 2]`` (static UEs) or ``[Q', K, 2]`` (one snapshot of
            the ``K`` UEs per time instant; the paper divides the rate sum by ``Q``, so
            normally ``Q' = Q``).
        sensing_metric: ``"eq8"`` or ``"archived"``.
        log_base: base of the logarithm in Eq. (15).
        one_plus_snr: see :func:`user_rates`.
        min_distance: lower clip of all distances.
    """
    tx_arr, rx_arr = _as_points(tx), _as_points(rx)
    det = fim_determinant(tx_arr, rx_arr, targets, sensing_metric, min_distance)
    ue_arr = np.asarray(ues, dtype=np.float64)
    if ue_arr.ndim == 2:
        ue_arr = ue_arr[None]
    rates = user_rates(
        np.concatenate([tx_arr, rx_arr], axis=0), ue_arr, log_base, one_plus_snr, min_distance
    )  # [Q', K]
    per_user = rates.mean(axis=0)  # R_k averaged over the instants
    return ObjectiveBreakdown(
        comm_sum=float(per_user.sum()),
        comm_min=float(per_user.min()),
        sens_mean=float(det.mean()),
        sens_min=float(det.min()),
    )
