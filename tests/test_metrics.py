"""Equation-level tests of the sensing and communication metrics (Section 2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from cfisac import reference
from cfisac.metrics import (
    ObjectiveBreakdown,
    bearing_angles,
    combine_fim_terms,
    evaluate_deployment,
    fim_determinant,
    fim_determinant_from_angles,
    fim_terms,
    snr,
    unit_directions,
    user_rates,
)


def _random_scene(rng, m, n, q=5):
    tx = rng.uniform(0, 50, size=(m, 2))
    rx = rng.uniform(0, 50, size=(n, 2))
    targets = rng.uniform(60, 90, size=(q, 2))  # disjoint from the APs
    return tx, rx, targets


# ------------------------------------------------------------------ Eq. (8)
def test_unit_directions_point_from_ap_to_target():
    dirs = unit_directions([[0.0, 0.0], [3.0, 0.0]], [[3.0, 4.0]])
    assert dirs.shape == (1, 2, 2)
    np.testing.assert_allclose(dirs[0, 0], [0.6, 0.8])
    np.testing.assert_allclose(dirs[0, 1], [0.0, 1.0])


def test_eq8_cross_geometry_closed_form():
    # Target at the origin; Tx west and east, Rx south and north.  The four rows of J0 are
    # (1,1), (1,-1), (-1,1), (-1,-1): A = B = 4, C = 0, so |Phi| = 16.
    tx = [[-7.0, 0.0], [3.0, 0.0]]
    rx = [[0.0, -2.0], [0.0, 9.0]]
    a, b, c = fim_terms(tx, rx, [[0.0, 0.0]])
    np.testing.assert_allclose([a[0], b[0], c[0]], [4.0, 4.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(fim_determinant(tx, rx, [[0.0, 0.0]]), [16.0])


def test_eq8_one_tx_two_rx_closed_form():
    # Tx west of the target, Rx south and north: rows (1,1) and (1,-1) -> |Phi| = 4.
    det = fim_determinant([[-5.0, 0.0]], [[0.0, -1.0], [0.0, 2.0]], [[0.0, 0.0]])
    np.testing.assert_allclose(det, [4.0])


def test_eq8_one_tx_two_rx_angle_formula(rng):
    # For M = 1, N = 2 the determinant is the squared cross product of the two rows:
    # (sin(t2 - tt) - sin(t1 - tt) + sin(t2 - t1))^2.
    for _ in range(20):
        tt, t1, t2 = rng.uniform(-np.pi, np.pi, size=3)
        expected = (math.sin(t2 - tt) - math.sin(t1 - tt) + math.sin(t2 - t1)) ** 2
        assert fim_determinant_from_angles([tt], [t1, t2]) == pytest.approx(expected, abs=1e-12)


def test_eq8_single_pair_is_identically_zero(rng):
    # One Tx and one Rx give a rank-one FIM (one bistatic range cannot fix a 2-D position).
    for _ in range(10):
        tx, rx, targets = _random_scene(rng, 1, 1)
        np.testing.assert_allclose(fim_determinant(tx, rx, targets), 0.0, atol=1e-12)


def test_eq8_colocated_aps_give_zero(rng):
    site = np.array([[12.0, 7.0]])
    targets = rng.uniform(20, 40, size=(6, 2))
    det = fim_determinant(np.repeat(site, 3, 0), np.repeat(site, 2, 0), targets)
    np.testing.assert_allclose(det, 0.0, atol=1e-9)


def test_eq8_bounds(rng):
    # 0 <= |Phi| <= 4 (M N)^2 because A + B = sum ||row||^2 <= 4 M N.
    for m, n in ((1, 2), (2, 2), (3, 4)):
        tx, rx, targets = _random_scene(rng, m, n, q=50)
        det = fim_determinant(tx, rx, targets)
        assert np.all(det >= -1e-9)
        assert np.all(det <= 4.0 * (m * n) ** 2 + 1e-9)


def test_eq8_matches_angular_form_eq7(rng):
    tx, rx, targets = _random_scene(rng, 3, 2)
    det = fim_determinant(tx, rx, targets)
    theta_t, theta_r = bearing_angles(tx, targets), bearing_angles(rx, targets)
    for q in range(len(targets)):
        assert det[q] == pytest.approx(fim_determinant_from_angles(theta_t[q], theta_r[q]))


def test_eq8_invariant_to_rigid_motion_and_scaling(rng):
    tx, rx, targets = _random_scene(rng, 2, 3)
    base = fim_determinant(tx, rx, targets)
    angle = 0.7
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    shift = np.array([13.0, -4.0])

    def move(points):
        return points @ rot.T + shift

    np.testing.assert_allclose(fim_determinant(move(tx), move(rx), move(targets)), base)
    np.testing.assert_allclose(fim_determinant(3.0 * tx, 3.0 * rx, 3.0 * targets), base)


def test_eq8_symmetric_in_ap_order_and_role_swap(rng):
    tx, rx, targets = _random_scene(rng, 3, 2)
    base = fim_determinant(tx, rx, targets)
    np.testing.assert_allclose(fim_determinant(tx[::-1], rx, targets), base)
    np.testing.assert_allclose(fim_determinant(tx, rx[[1, 0]], targets), base)
    np.testing.assert_allclose(fim_determinant(rx, tx, targets), base)


def test_vectorised_matches_reference_and_lagrange(rng):
    for m, n in ((1, 1), (1, 3), (2, 2), (4, 3)):
        tx, rx, targets = _random_scene(rng, m, n)
        det = fim_determinant(tx, rx, targets)
        for q, p in enumerate(targets):
            ref = reference.fim_determinant_reference(tx.tolist(), rx.tolist(), p.tolist())
            lag = reference.fim_determinant_lagrange(tx.tolist(), rx.tolist(), p.tolist())
            assert det[q] == pytest.approx(ref, rel=1e-10, abs=1e-10)
            assert det[q] == pytest.approx(lag, rel=1e-9, abs=1e-9)


def test_archived_metric_is_a_times_b_minus_c(rng):
    tx, rx, targets = _random_scene(rng, 2, 2)
    a, b, c = fim_terms(tx, rx, targets)
    np.testing.assert_allclose(fim_determinant(tx, rx, targets, "archived"), a * b - c)
    np.testing.assert_allclose(combine_fim_terms(a, b, c, "eq8"), a * b - c**2)
    with pytest.raises(ValueError):
        combine_fim_terms(a, b, c, "nope")


def test_archived_metric_nonzero_for_colocated_pair():
    # Unlike Eq. (8), the archived quantity does not vanish for a single co-located pair:
    # A B - C = 16 cos^2 sin^2 - 4 cos sin, e.g. 4 + 2 = 6 at a bearing of 135 degrees.
    det = fim_determinant([[1.0, 0.0]], [[1.0, 0.0]], [[0.0, 1.0]], "archived")
    np.testing.assert_allclose(det, [6.0])


def test_ap_on_target_point_is_finite():
    det = fim_determinant([[0.0, 0.0], [1.0, 1.0]], [[2.0, 0.0], [0.0, 3.0]], [[0.0, 0.0]])
    assert np.all(np.isfinite(det))


# -------------------------------------------------------------- Eqs. (14)-(15)
def test_snr_eq14_hand_values():
    aps = [[0.0, 0.0], [0.0, 10.0]]
    ues = [[3.0, 4.0], [0.0, 5.0]]
    expected = [1 / 25 + 1 / (9 + 36), 1 / 25 + 1 / 25]
    np.testing.assert_allclose(snr(aps, ues), expected)


def test_snr_batched_shapes(rng):
    aps = rng.uniform(0, 50, size=(4, 2))
    ues = rng.uniform(0, 50, size=(7, 3, 2))
    out = snr(aps, ues)
    assert out.shape == (7, 3)
    for q in range(7):
        for k in range(3):
            assert out[q, k] == pytest.approx(reference.snr_reference(aps.tolist(), ues[q, k]))


def test_rate_eq15_forms():
    aps, ues = [[0.0, 0.0]], [[2.0, 0.0]]  # SNR = 0.25
    assert user_rates(aps, ues, log_base=10.0)[0] == pytest.approx(math.log10(1.25))
    assert user_rates(aps, ues, log_base=2.0)[0] == pytest.approx(math.log2(1.25))
    # Literal last line of Eq. (15), without the "1 +".
    assert user_rates(aps, ues, one_plus_snr=False)[0] == pytest.approx(math.log10(0.25))


def test_ap_on_ue_is_finite():
    assert np.isfinite(user_rates([[1.0, 1.0]], [[1.0, 1.0]])).all()


# ------------------------------------------------------------------- Eq. (17)
def test_objective_breakdown_composition():
    b = ObjectiveBreakdown(comm_sum=2.0, comm_min=0.5, sens_mean=10.0, sens_min=4.0)
    assert b.isac("max-sum") == pytest.approx(20.0)
    assert b.isac("max-min") == pytest.approx(2.0)
    assert b.value("max-sum", "sensing") == 10.0
    assert b.value("max-min", "communication") == 0.5
    assert b.value("max-sum", "weighted", weight=0.25) == pytest.approx(0.25 * 2 + 0.75 * 10)
    with pytest.raises(ValueError):
        b.value("max-avg")
    with pytest.raises(ValueError):
        b.value("max-sum", "other")


def test_evaluate_deployment_hand_case():
    # One trajectory point at the origin with the cross geometry (|Phi| = 16) and two static
    # UEs; all four APs are at distance 5 from UE 1.
    tx = [[-5.0, 0.0], [5.0, 0.0]]
    rx = [[0.0, -5.0], [0.0, 5.0]]
    ues = [[0.0, 0.0], [0.0, 10.0]]
    b = evaluate_deployment(tx, rx, [[0.0, 0.0]], ues, log_base=10.0)
    r1 = math.log10(1 + 4 / 25)
    r2 = math.log10(1 + 1 / 125 + 1 / 125 + 1 / 225 + 1 / 25)
    assert b.sens_mean == pytest.approx(16.0) and b.sens_min == pytest.approx(16.0)
    assert b.comm_sum == pytest.approx(r1 + r2)
    assert b.comm_min == pytest.approx(min(r1, r2))


@pytest.mark.parametrize("aggregation", ["max-sum", "max-min"])
def test_objective_matches_reference(rng, aggregation):
    tx, rx, targets = _random_scene(rng, 2, 2, q=6)
    ues = rng.uniform(0, 50, size=(6, 3, 2))
    value = evaluate_deployment(tx, rx, targets, ues).isac(aggregation)
    ref = reference.objective_reference(
        tx.tolist(), rx.tolist(), targets.tolist(), ues.tolist(), aggregation
    )
    assert value == pytest.approx(ref, rel=1e-10)
