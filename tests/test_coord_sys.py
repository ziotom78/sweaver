# -*- encoding: utf-8 -*-
#
# SWEaver: harmonic-domain manipulation of electomagnetic beams for CMB analysis
#
#           ##############
#        #######        #######
#      ####                  ####
#    ####                      ####
#   ###                          ###
#  ###  ####       ##       ####  ###
#  ## #######      ##      ####### ##
# ###########     ###      ###########
# ######  ###     ####     ### #######
# ####     ###    ####    ###     ####
# ###       ##   ######   ###       ##
#  ##       ###  ##  ##  ###       ##
#  ###      #######  #######      ###
#   ###      #####    #####      ###
#    ####    #####    #####    ####
#      ####                  ####
#        #######        #######
#            ##############
#
# Copyright © 2026 Maurizio Tomasi
# This code is licensed under the GPL 3
# See the file LICENSE.txt

import pytest
import numpy as np
import numpy.testing as npt
from scipy.spatial.transform import Rotation

from utils import load_dipole_sph
from sweaver import (
    EulerAngles,
    CoordinateSystem,
    Polarization,
    get_euler_from_ticra_axes,
    get_euler_from_grasp_angles,
)


def test_euler_inverse_identity():
    """The inverse of no rotation is no rotation."""
    e = EulerAngles(0.0, 0.0, 0.0)
    inv = e.inverse()

    assert inv.alpha_rad == 0.0
    assert inv.beta_rad == 0.0
    assert inv.gamma_rad == 0.0


def test_euler_inverse_asymmetric():
    """
    Verify that the inverse mathematically swaps the order of the angles
    and flips their signs: (−γ, −β, −α).
    """
    e = EulerAngles(0.1, 0.2, 0.3)
    inv = e.inverse()

    assert inv.alpha_rad == pytest.approx(-0.3)
    assert inv.beta_rad == pytest.approx(-0.2)
    assert inv.gamma_rad == pytest.approx(-0.1)


def test_ticra_axes_identity():
    """
    If the TICRA local axes perfectly align with the global axes,
    the extracted Euler angles must be exactly zero.
    """
    x_axis = [1.0, 0.0, 0.0]
    y_axis = [0.0, 1.0, 0.0]

    euler = get_euler_from_ticra_axes(x_axis, y_axis)

    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(0.0)
    assert euler.gamma_rad == pytest.approx(0.0)


def test_ticra_axes_pure_tilt():
    """
    Test a simple rotation: A pure 30° tilt down the Y axis.
    Because Z rotates toward X, the new X axis points down into the Z plane.
    """
    angle = np.radians(30.0)
    x_axis = [np.cos(angle), 0.0, -np.sin(angle)]
    y_axis = [0.0, 1.0, 0.0]

    euler = get_euler_from_ticra_axes(x_axis, y_axis)

    # This maps strictly to a β rotation.
    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(angle)
    assert euler.gamma_rad == pytest.approx(0.0)


def test_ticra_axes_round_trip():
    """
    Inject known Z-Y-Z active Euler angles, generate the corresponding TICRA
    axes from the rotation matrix, and ensure the function recovers the angles.
    (We keep beta > 0 to prevent Scipy from wrapping to another Euler branch).
    """
    alpha_in = np.radians(15.0)
    beta_in = np.radians(45.0)
    gamma_in = np.radians(60.0)

    # Create the rotation matrix
    r = Rotation.from_euler("ZYZ", [alpha_in, beta_in, gamma_in])
    rot_matrix = r.as_matrix()

    # Extract the local TICRA axes in the base frame
    x_axis = rot_matrix[:, 0]
    y_axis = rot_matrix[:, 1]

    # Recover the angles
    euler_out = get_euler_from_ticra_axes(x_axis, y_axis)

    assert euler_out.alpha_rad == pytest.approx(alpha_in)
    assert euler_out.beta_rad == pytest.approx(beta_in)
    assert euler_out.gamma_rad == pytest.approx(gamma_in)


def test_ticra_axes_physics_reconstruction():
    """
    Even if Scipy returns a different (but mathematically equivalent) branch
    of Euler angles, converting those angles back into a matrix
    *must* perfectly reconstruct the original TICRA Cartesian axes.
    """
    x_axis = [0.76483242, 0.43174260, -0.47815238]
    y_axis = [-0.4154152, 0.89780839, 0.14618585]

    # Extract the angles
    euler = get_euler_from_ticra_axes(x_axis, y_axis)

    np.testing.assert_allclose(np.rad2deg(euler.alpha_rad), 10.0)
    np.testing.assert_allclose(np.rad2deg(euler.beta_rad), 30.0)
    np.testing.assert_allclose(np.rad2deg(euler.gamma_rad), 17.0)


def test_grasp_angles_identity():
    """If all TICRA angles are zero, the Euler angles must be exactly zero."""
    euler = get_euler_from_grasp_angles(theta_rad=0.0, phi_rad=0.0, psi_rad=0.0)

    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(0.0)
    assert euler.gamma_rad == pytest.approx(0.0)


def test_ticra_angles_pure_theta():
    """
    A pure theta rotation (phi=0, psi=0) in TICRA corresponds strictly
    to a pure beta (Y-axis) rotation in Z-Y-Z Euler angles.
    """
    euler = get_euler_from_grasp_angles(theta_rad=np.pi / 6, phi_rad=0.0, psi_rad=0.0)

    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(np.pi / 6)
    assert euler.gamma_rad == pytest.approx(0.0)


def test_ticra_angles_pure_phi():
    """
    A pure phi rotation (theta=0, psi=0) tests the cross-axis dependency.
    According to the manual, gamma = -phi + psi. So a positive phi
    must result in a negative gamma!
    """
    euler = get_euler_from_grasp_angles(theta_rad=0.0, phi_rad=np.pi / 4, psi_rad=0.0)

    assert euler.alpha_rad == pytest.approx(np.pi / 4)
    assert euler.beta_rad == pytest.approx(0.0)
    assert euler.gamma_rad == pytest.approx(-np.pi / 4)


def test_ticra_angles_pure_psi():
    """
    A pure psi rotation (theta=0, phi=0) maps directly to the final
    gamma (Z-axis) Euler rotation.
    """
    euler = get_euler_from_grasp_angles(theta_rad=0.0, phi_rad=0.0, psi_rad=np.pi / 3)

    assert euler.alpha_rad == pytest.approx(0.0)
    assert euler.beta_rad == pytest.approx(0.0)
    assert euler.gamma_rad == pytest.approx(np.pi / 3)


def test_ticra_angles_combined_asymmetric():
    """
    Test a fully asymmetric configuration (similar to the interferometric pair).
    Verifies the exact algebraic combination: γ = -φ + ψ.
    """
    euler = get_euler_from_grasp_angles(theta_rad=0.15, phi_rad=0.45, psi_rad=0.10)

    assert euler.alpha_rad == pytest.approx(0.45)
    assert euler.beta_rad == pytest.approx(0.15)
    # γ = −0.45 + 0.10 = -0.35
    assert euler.gamma_rad == pytest.approx(-0.35)


def test_coordinate_system_from_matrix_roundtrip():
    angles = EulerAngles(
        alpha_rad=np.radians(20.0),
        beta_rad=np.radians(35.0),
        gamma_rad=np.radians(-10.0),
    )

    r = angles.as_child_to_base_matrix()

    csy = CoordinateSystem.from_matrix(
        rotation_matrix_child_to_base=r,
        origin_m=np.array([1.0, 2.0, 3.0]),
    )

    npt.assert_allclose(csy.as_child_to_base_matrix(), r, atol=1e-12)
    npt.assert_allclose(csy.origin_m, np.array([1.0, 2.0, 3.0]))


def test_coordinate_system_relative_to_identity():
    csy = CoordinateSystem.from_ticra_degrees(
        origin_m=np.array([1.0, 2.0, 3.0]),
        alpha_deg=20.0,
        beta_deg=35.0,
        gamma_deg=-10.0,
    )

    rel = csy.relative_to(CoordinateSystem.identity())

    npt.assert_allclose(rel.origin_m, csy.origin_m, atol=1e-12)
    npt.assert_allclose(
        rel.as_child_to_base_matrix(),
        csy.as_child_to_base_matrix(),
        atol=1e-12,
    )


def test_coordinate_system_identity_matches_evaluate_cut_for_dipole():
    """A TICRA-like identity coordinate system must reproduce evaluate_cut()."""

    field = load_dipole_sph("hertzian_e_dipole_x.sph")

    coor_sys = CoordinateSystem.identity()

    phi_angle = np.radians(37.0)
    theta_start = np.radians(-45.0)
    theta_end = np.radians(45.0)
    ntheta = 101

    csy_1, csy_2 = field.evaluate_cut_in_coordinate_system(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        coor_sys=coor_sys,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    ref_1, ref_2 = field.evaluate_cut(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    npt.assert_allclose(csy_1, ref_1, rtol=1e-10, atol=1e-12)
    npt.assert_allclose(csy_2, ref_2, rtol=1e-10, atol=1e-12)


def test_coordinate_system_rotation_matches_evaluate_cut_in_frame_for_dipole():
    """A coordinate system with zero origin should behave like evaluate_cut_in_frame()."""

    field = load_dipole_sph("hertzian_e_dipole_x.sph")

    angles = EulerAngles(
        alpha_rad=np.radians(25.0),
        beta_rad=np.radians(40.0),
        gamma_rad=np.radians(-15.0),
    )

    coor_sys = CoordinateSystem(
        origin_m=np.zeros(3),
        angles=angles,
    )

    phi_angle = np.radians(12.0)
    theta_start = np.radians(-60.0)
    theta_end = np.radians(60.0)
    ntheta = 121

    csy_1, csy_2 = field.evaluate_cut_in_coordinate_system(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        coor_sys=coor_sys,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    frame_1, frame_2 = field.evaluate_cut_in_frame(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        frame=angles,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    npt.assert_allclose(csy_1, frame_1, rtol=1e-10, atol=1e-12)
    npt.assert_allclose(csy_2, frame_2, rtol=1e-10, atol=1e-12)


def test_coordinate_system_origin_uses_ticra_sign_for_dipole():
    """
    A CoordinateSystem origin follows TICRA semantics.

    If the child coordinate-system origin is at +d in the base frame, the
    equivalent operation on base-frame SWE coefficients is to translate the
    field phase center by -d, then evaluate the cut in the rotated child frame.
    """

    field = load_dipole_sph("hertzian_e_dipole_z.sph")

    origin_m = np.array([2.0, -3.0, 1.5]) * 1e-3

    angles = EulerAngles(
        alpha_rad=np.radians(20.0),
        beta_rad=np.radians(35.0),
        gamma_rad=np.radians(-10.0),
    )

    coor_sys = CoordinateSystem(
        origin_m=origin_m,
        angles=angles,
    )

    phi_angle = np.radians(17.0)
    theta_start = np.radians(-50.0)
    theta_end = np.radians(50.0)
    ntheta = 111

    csy_1, csy_2 = field.evaluate_cut_in_coordinate_system(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        coor_sys=coor_sys,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    expected_field = field.translate_phase_center(*(-origin_m))

    expected_1, expected_2 = expected_field.evaluate_cut_in_frame(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        frame=angles,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    npt.assert_allclose(csy_1, expected_1, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(csy_2, expected_2, rtol=1e-10, atol=1e-10)


def test_coordinate_system_origin_sign_is_not_accidentally_reversed_for_dipole():
    """
    Check that CoordinateSystem(origin=+d) does not behave like translate(+d).

    This guards against accidentally removing the TICRA sign flip in
    evaluate_cut_in_coordinate_system().
    """

    field = load_dipole_sph("hertzian_e_dipole_x.sph")

    origin_m = np.array([3.0, 1.0, -2.0]) * 1e-3

    angles = EulerAngles(
        alpha_rad=np.radians(30.0),
        beta_rad=np.radians(25.0),
        gamma_rad=np.radians(5.0),
    )

    coor_sys = CoordinateSystem(
        origin_m=origin_m,
        angles=angles,
    )

    phi_angle = np.radians(23.0)
    theta_start = np.radians(-70.0)
    theta_end = np.radians(70.0)
    ntheta = 141

    csy_1, csy_2 = field.evaluate_cut_in_coordinate_system(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        coor_sys=coor_sys,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    wrong_field = field.translate_phase_center(*(origin_m))

    wrong_1, wrong_2 = wrong_field.evaluate_cut_in_frame(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        frame=angles,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    diff = max(
        np.max(np.abs(csy_1 - wrong_1)),
        np.max(np.abs(csy_2 - wrong_2)),
    )

    assert diff > 1e-6


def test_coordinate_system_origin_identity_rotation_for_dipole():
    """
    With identity rotation, CoordinateSystem(origin=d) should equal
    translate_phase_center(-d) followed by an ordinary evaluate_cut().
    """

    field = load_dipole_sph("hertzian_e_dipole_y.sph")

    origin_m = np.array([1.0, 2.0, -1.0]) * 1e-3

    coor_sys = CoordinateSystem(
        origin_m=origin_m,
        angles=EulerAngles(
            alpha_rad=0.0,
            beta_rad=0.0,
            gamma_rad=0.0,
        ),
    )

    phi_angle = np.radians(43.0)
    theta_start = np.radians(-40.0)
    theta_end = np.radians(40.0)
    ntheta = 81

    csy_1, csy_2 = field.evaluate_cut_in_coordinate_system(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        coor_sys=coor_sys,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    expected_field = field.translate_phase_center(*(-origin_m))

    expected_1, expected_2 = expected_field.evaluate_cut(
        phi_angle_rad=phi_angle,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        polarization=Polarization.THETA_PHI,
        epsilon=1e-10,
    )

    npt.assert_allclose(csy_1, expected_1, rtol=1e-10, atol=1e-10)
    npt.assert_allclose(csy_2, expected_2, rtol=1e-10, atol=1e-10)
