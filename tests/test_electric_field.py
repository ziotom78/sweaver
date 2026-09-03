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

import gzip

import numpy as np
import numpy.testing as npt
import pytest

import sweaver
from utils import load_grd_file, load_dipole_sph, load_asymmetric_field


@pytest.mark.parametrize(
    "data_file,expected_header,polarization",
    [
        (
            "asymmetric_grid_thetaphi.grd.gz",
            (1, 1, 2, 7),
            sweaver.Polarization.THETA_PHI,
        ),
        ("asymmetric_grid_ludwig.grd.gz", (1, 3, 2, 7), sweaver.Polarization.LUDWIG3_X),
    ],
)
def test_asymmetric_field(
    data_file: str,
    expected_header: tuple[int, int, int, int],
    polarization: sweaver.Polarization,
) -> None:
    # Here we read a “hard” pattern: an asymmetric main lobe with diffraction
    # sidelobes produced by a rectangular hor

    with gzip.open(sweaver.get_test_data_path("asymmetric_swe.sph.gz"), "rt") as f:
        grasp_file = sweaver.read_sph_file(f)
    assert grasp_file.num_of_blocks == 1

    electric_field = sweaver.ElectricField.from_frequency_block(grasp_file.get(index=0))

    with gzip.open(sweaver.get_test_data_path(data_file), "rt") as f:
        grid = load_grd_file(f, expected_header=expected_header)

    e_theta, e_phi = electric_field.evaluate_theta_phi_grid(
        theta_start_rad=grid.theta_start_rad,
        theta_end_rad=grid.theta_end_rad,
        ntheta=grid.ntheta,
        phi_start_rad=grid.phi_start_rad,
        phi_end_rad=grid.phi_end_rad,
        nphi=grid.nphi,
        polarization=polarization,
        epsilon=1e-9,
    )

    rel_differences = np.concatenate(
        [
            np.abs(e_theta.flatten() - grid.e_field[:, 0]),
            np.abs(e_phi.flatten() - grid.e_field[:, 1]),
        ]
    )

    # We do not aim to a better value than 10⁻⁵ because
    # the highest multipoles  in the `.sph` file produced by GRASP has
    # still ~10⁻⁹ power right before the truncation. As the beam has an
    # overall power of ~44 W, this means that the relative error in power
    # is ~10⁻¹⁰, and thus the error in amplitude is of the order of 10⁻⁵.
    assert np.median(rel_differences) < 1.0e-5, "The error on the E field is too large"

    npt.assert_allclose(e_theta.flatten(), grid.e_field[:, 0], atol=1e-4)
    npt.assert_allclose(e_phi.flatten(), grid.e_field[:, 1], atol=1e-4)


@pytest.fixture(scope="module")
def asymmetric_field() -> sweaver.ElectricField:
    return load_asymmetric_field()


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def rotation_matrix_y(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ]
    )


def rotation_matrix_x(angle_rad: float) -> np.ndarray:
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ]
    )


def sph_to_cart(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    return np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=0,
    )


def cart_to_sph(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = n

    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.mod(np.arctan2(y, x), 2 * np.pi)

    return theta, phi


def ludwig3_x_from_theta_phi(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        e_theta * np.cos(phi) - e_phi * np.sin(phi),
        e_theta * np.sin(phi) + e_phi * np.cos(phi),
    )


def ludwig3_y_from_theta_phi(
    e_theta: np.ndarray,
    e_phi: np.ndarray,
    phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return (
        e_theta * np.sin(phi) + e_phi * np.cos(phi),
        e_theta * np.cos(phi) - e_phi * np.sin(phi),
    )


def test_rotation_consistency():
    field_x = load_dipole_sph("hertzian_e_dipole_x.sph")
    field_y = load_dipole_sph("hertzian_e_dipole_y.sph")
    field_z = load_dipole_sph("hertzian_e_dipole_z.sph")

    rotated_x = field_x.rotate_euler(
        sweaver.EulerAngles(
            alpha_rad=np.pi / 2,
            beta_rad=0.0,
            gamma_rad=0.0,
        )
    )

    rotated_x_cmp, field_y_cmp = rotated_x.compatible_with(field_y)

    assert np.allclose(
        rotated_x_cmp.alm_stack,
        field_y_cmp.alm_stack,
        atol=1e-12,
    )

    rotated_z = field_z.rotate_euler(
        sweaver.EulerAngles(
            alpha_rad=0.0,
            beta_rad=np.pi / 2,
            gamma_rad=0.0,
        )
    )

    rotated_z_cmp, field_x_cmp = rotated_z.compatible_with(field_x)

    assert np.allclose(
        rotated_z_cmp.alm_stack,
        field_x_cmp.alm_stack,
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "polarization",
    [
        sweaver.Polarization.THETA_PHI,
        sweaver.Polarization.LUDWIG3_X,
        sweaver.Polarization.LUDWIG3_Y,
    ],
)
def test_evaluate_at_locs_matches_theta_phi_grid(
    asymmetric_field: sweaver.ElectricField,
    polarization: sweaver.Polarization,
) -> None:
    theta_start = np.radians(2.0)
    theta_end = np.radians(11.0)
    ntheta = 8

    phi_start = np.radians(15.0)
    phi_end = np.radians(72.0)
    nphi = 7

    theta = np.linspace(theta_start, theta_end, ntheta)
    phi = np.linspace(phi_start, phi_end, nphi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    loc_1, loc_2 = asymmetric_field.evaluate_at_locs(
        theta_rad=theta_grid,
        phi_rad=phi_grid,
        polarization=polarization,
        epsilon=1e-9,
    )

    grid_1, grid_2 = asymmetric_field.evaluate_theta_phi_grid(
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        phi_start_rad=phi_start,
        phi_end_rad=phi_end,
        nphi=nphi,
        polarization=polarization,
        epsilon=1e-9,
    )

    npt.assert_allclose(loc_1, grid_1, rtol=0.0, atol=1e-13)
    npt.assert_allclose(loc_2, grid_2, rtol=0.0, atol=1e-13)

    def test_evaluate_at_locs_preserves_input_shape(
        asymmetric_field: sweaver.ElectricField,
    ) -> None:
        theta = np.array(
            [
                [0.10, 0.20, 0.30],
                [0.40, 0.50, 0.60],
            ]
        )

        phi = np.array(
            [
                [0.00, 0.25, 0.50],
                [0.75, 1.00, 1.25],
            ]
        )

        e_theta, e_phi = asymmetric_field.evaluate_at_locs(
            theta_rad=theta,
            phi_rad=phi,
            polarization=sweaver.Polarization.THETA_PHI,
        )

        assert e_theta.shape == theta.shape
        assert e_phi.shape == theta.shape


def test_evaluate_at_locs_rejects_mismatched_shapes(
    asymmetric_field: sweaver.ElectricField,
) -> None:
    theta = np.zeros((2, 3))
    phi = np.zeros((2, 4))

    with pytest.raises(ValueError, match="same shape"):
        asymmetric_field.evaluate_at_locs(
            theta_rad=theta,
            phi_rad=phi,
            polarization=sweaver.Polarization.THETA_PHI,
        )


def test_evaluate_at_locs_negative_theta_convention(
    asymmetric_field: sweaver.ElectricField,
) -> None:
    theta = np.radians(np.array([3.0, 7.0, 11.0, 15.0]))
    phi = np.radians(np.array([10.0, 50.0, 130.0, 250.0]))

    e_theta_neg, e_phi_neg = asymmetric_field.evaluate_at_locs(
        theta_rad=-theta,
        phi_rad=phi,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    e_theta_ref, e_phi_ref = asymmetric_field.evaluate_at_locs(
        theta_rad=theta,
        phi_rad=phi + np.pi,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    npt.assert_allclose(e_theta_neg, -e_theta_ref, rtol=0.0, atol=1e-12)
    npt.assert_allclose(e_phi_neg, -e_phi_ref, rtol=0.0, atol=1e-12)

    @pytest.mark.parametrize(
        "polarization",
        [
            sweaver.Polarization.THETA_PHI,
            sweaver.Polarization.LUDWIG3_X,
            sweaver.Polarization.LUDWIG3_Y,
        ],
    )
    def test_evaluate_at_locs_ticra_phase_is_conjugation(
        asymmetric_field: sweaver.ElectricField,
        polarization: sweaver.Polarization,
    ) -> None:
        theta = np.radians(np.array([4.0, 8.0, 12.0]))
        phi = np.radians(np.array([20.0, 70.0, 140.0]))

        e1, e2 = asymmetric_field.evaluate_at_locs(
            theta_rad=theta,
            phi_rad=phi,
            polarization=polarization,
            epsilon=1e-9,
            use_ticra_phase=False,
        )

        e1_ticra, e2_ticra = asymmetric_field.evaluate_at_locs(
            theta_rad=theta,
            phi_rad=phi,
            polarization=polarization,
            epsilon=1e-9,
            use_ticra_phase=True,
        )

        npt.assert_allclose(e1_ticra, np.conj(e1), rtol=0.0, atol=1e-13)
        npt.assert_allclose(e2_ticra, np.conj(e2), rtol=0.0, atol=1e-13)


@pytest.mark.parametrize(
    "polarization,manual_projection",
    [
        (sweaver.Polarization.LUDWIG3_X, ludwig3_x_from_theta_phi),
        (sweaver.Polarization.LUDWIG3_Y, ludwig3_y_from_theta_phi),
    ],
)
def test_evaluate_at_locs_ludwig_projection_matches_manual_formula(
    asymmetric_field: sweaver.ElectricField,
    polarization: sweaver.Polarization,
    manual_projection,
) -> None:
    theta = np.radians(np.array([5.0, 9.0, 13.0, 17.0]))
    phi = np.radians(np.array([0.0, 35.0, 80.0, 155.0]))

    e_theta, e_phi = asymmetric_field.evaluate_at_locs(
        theta_rad=theta,
        phi_rad=phi,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    expected_1, expected_2 = manual_projection(e_theta, e_phi, phi)

    actual_1, actual_2 = asymmetric_field.evaluate_at_locs(
        theta_rad=theta,
        phi_rad=phi,
        polarization=polarization,
        epsilon=1e-9,
    )

    npt.assert_allclose(actual_1, expected_1, rtol=0.0, atol=1e-13)
    npt.assert_allclose(actual_2, expected_2, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize(
    "polarization",
    [
        sweaver.Polarization.THETA_PHI,
        sweaver.Polarization.LUDWIG3_X,
        sweaver.Polarization.LUDWIG3_Y,
    ],
)
def test_evaluate_cut_matches_single_phi_grid(
    asymmetric_field: sweaver.ElectricField,
    polarization: sweaver.Polarization,
) -> None:
    phi = np.radians(37.0)

    theta_start = np.radians(-12.0)
    theta_end = np.radians(18.0)
    ntheta = 31

    cut_1, cut_2 = asymmetric_field.evaluate_cut(
        phi_angle_rad=phi,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        polarization=polarization,
        epsilon=1e-9,
    )

    grid_1, grid_2 = asymmetric_field.evaluate_theta_phi_grid(
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        phi_start_rad=phi,
        phi_end_rad=phi,
        nphi=1,
        polarization=polarization,
        epsilon=1e-9,
    )

    npt.assert_allclose(cut_1, grid_1[:, 0], rtol=0.0, atol=1e-13)
    npt.assert_allclose(cut_2, grid_2[:, 0], rtol=0.0, atol=1e-13)


@pytest.mark.parametrize(
    "polarization",
    [
        sweaver.Polarization.THETA_PHI,
        sweaver.Polarization.LUDWIG3_X,
        sweaver.Polarization.LUDWIG3_Y,
    ],
)
def test_evaluate_cut_matches_evaluate_at_locs(
    asymmetric_field: sweaver.ElectricField,
    polarization: sweaver.Polarization,
) -> None:
    phi = np.radians(123.0)
    theta = np.linspace(np.radians(-20.0), np.radians(25.0), 46)
    phi_array = np.full_like(theta, phi)

    cut_1, cut_2 = asymmetric_field.evaluate_cut(
        phi_angle_rad=phi,
        theta_start_rad=theta[0],
        theta_end_rad=theta[-1],
        ntheta=theta.size,
        polarization=polarization,
        epsilon=1e-9,
    )

    loc_1, loc_2 = asymmetric_field.evaluate_at_locs(
        theta_rad=theta,
        phi_rad=phi_array,
        polarization=polarization,
        epsilon=1e-9,
    )

    npt.assert_allclose(cut_1, loc_1, rtol=0.0, atol=1e-13)
    npt.assert_allclose(cut_2, loc_2, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize(
    "polarization",
    [
        sweaver.Polarization.THETA_PHI,
        sweaver.Polarization.LUDWIG3_X,
        sweaver.Polarization.LUDWIG3_Y,
    ],
)
def test_evaluate_in_frame_identity_matches_evaluate_at_locs(
    asymmetric_field: sweaver.ElectricField,
    polarization: sweaver.Polarization,
) -> None:
    theta = np.radians(
        np.array(
            [
                [2.0, 4.0, 6.0],
                [8.0, 10.0, 12.0],
            ]
        )
    )
    phi = np.radians(
        np.array(
            [
                [0.0, 30.0, 60.0],
                [90.0, 120.0, 150.0],
            ]
        )
    )

    in_frame_1, in_frame_2 = asymmetric_field.evaluate_in_frame(
        theta_rad=theta,
        phi_rad=phi,
        rotation_matrix_child_to_base=np.eye(3),
        polarization=polarization,
        epsilon=1e-9,
    )

    loc_1, loc_2 = asymmetric_field.evaluate_at_locs(
        theta_rad=theta,
        phi_rad=phi,
        polarization=polarization,
        epsilon=1e-9,
    )

    npt.assert_allclose(in_frame_1, loc_1, rtol=1e-10, atol=1e-12)
    npt.assert_allclose(in_frame_2, loc_2, rtol=1e-10, atol=1e-12)


def test_evaluate_in_frame_rejects_mismatched_angle_shapes(
    asymmetric_field: sweaver.ElectricField,
) -> None:
    theta = np.zeros((2, 3))
    phi = np.zeros((3, 2))

    with pytest.raises(ValueError, match="same shape"):
        asymmetric_field.evaluate_in_frame(
            theta_rad=theta,
            phi_rad=phi,
            rotation_matrix_child_to_base=np.eye(3),
            polarization=sweaver.Polarization.THETA_PHI,
        )


@pytest.mark.parametrize(
    "bad_matrix",
    [
        np.zeros((2, 2)),
        np.zeros((3, 4)),
        np.zeros((4, 3)),
        np.zeros(3),
    ],
)
def test_evaluate_in_frame_rejects_bad_rotation_matrix(
    asymmetric_field: sweaver.ElectricField,
    bad_matrix: np.ndarray,
) -> None:
    theta = np.array([0.1, 0.2])
    phi = np.array([0.3, 0.4])

    with pytest.raises(ValueError, match="shape"):
        asymmetric_field.evaluate_in_frame(
            theta_rad=theta,
            phi_rad=phi,
            rotation_matrix_child_to_base=bad_matrix,
            polarization=sweaver.Polarization.THETA_PHI,
        )


def test_evaluate_in_frame_z_rotation_theta_phi_components(
    asymmetric_field: sweaver.ElectricField,
) -> None:
    delta = np.radians(37.0)
    R_child_to_base = rotation_matrix_z(delta)

    theta = np.radians(np.array([3.0, 8.0, 13.0, 18.0]))
    phi_child = np.radians(np.array([0.0, 25.0, 70.0, 140.0]))
    phi_base = np.mod(phi_child + delta, 2 * np.pi)

    e_theta_child, e_phi_child = asymmetric_field.evaluate_in_frame(
        theta_rad=theta,
        phi_rad=phi_child,
        rotation_matrix_child_to_base=R_child_to_base,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    e_theta_base, e_phi_base = asymmetric_field.evaluate_at_locs(
        theta_rad=theta,
        phi_rad=phi_base,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    npt.assert_allclose(e_theta_child, e_theta_base, rtol=0.0, atol=1e-12)
    npt.assert_allclose(e_phi_child, e_phi_base, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize(
    "polarization,manual_projection",
    [
        (sweaver.Polarization.LUDWIG3_X, ludwig3_x_from_theta_phi),
        (sweaver.Polarization.LUDWIG3_Y, ludwig3_y_from_theta_phi),
    ],
)
def test_evaluate_in_frame_z_rotation_ludwig_uses_child_phi(
    asymmetric_field: sweaver.ElectricField,
    polarization: sweaver.Polarization,
    manual_projection,
) -> None:
    delta = np.radians(41.0)
    R_child_to_base = rotation_matrix_z(delta)

    theta = np.radians(np.array([4.0, 9.0, 14.0]))
    phi_child = np.radians(np.array([10.0, 80.0, 170.0]))
    phi_base = np.mod(phi_child + delta, 2 * np.pi)

    # For a pure z-rotation, the theta/phi components in the child frame
    # equal the base-frame components sampled at phi_base.
    e_theta_base, e_phi_base = asymmetric_field.evaluate_at_locs(
        theta_rad=theta,
        phi_rad=phi_base,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    expected_1, expected_2 = manual_projection(
        e_theta_base,
        e_phi_base,
        phi_child,
    )

    actual_1, actual_2 = asymmetric_field.evaluate_in_frame(
        theta_rad=theta,
        phi_rad=phi_child,
        rotation_matrix_child_to_base=R_child_to_base,
        polarization=polarization,
        epsilon=1e-9,
    )

    npt.assert_allclose(actual_1, expected_1, rtol=0.0, atol=1e-12)
    npt.assert_allclose(actual_2, expected_2, rtol=0.0, atol=1e-12)


def test_evaluate_in_frame_arbitrary_rotation_preserves_intensity(
    asymmetric_field: sweaver.ElectricField,
) -> None:
    R_child_to_base = (
        rotation_matrix_z(np.radians(25.0))
        @ rotation_matrix_y(np.radians(17.0))
        @ rotation_matrix_x(np.radians(-11.0))
    )

    theta_child = np.radians(
        np.array(
            [
                [5.0, 10.0, 15.0],
                [20.0, 25.0, 30.0],
            ]
        )
    )

    phi_child = np.radians(
        np.array(
            [
                [0.0, 35.0, 80.0],
                [125.0, 210.0, 300.0],
            ]
        )
    )

    e_theta_child, e_phi_child = asymmetric_field.evaluate_in_frame(
        theta_rad=theta_child,
        phi_rad=phi_child,
        rotation_matrix_child_to_base=R_child_to_base,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    n_child = sph_to_cart(theta_child, phi_child)
    n_base = np.einsum("ij,j...->i...", R_child_to_base, n_child)
    theta_base, phi_base = cart_to_sph(n_base)

    e_theta_base, e_phi_base = asymmetric_field.evaluate_at_locs(
        theta_rad=theta_base,
        phi_rad=phi_base,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    intensity_child = np.abs(e_theta_child) ** 2 + np.abs(e_phi_child) ** 2
    intensity_base = np.abs(e_theta_base) ** 2 + np.abs(e_phi_base) ** 2

    npt.assert_allclose(intensity_child, intensity_base, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize(
    "polarization",
    [
        sweaver.Polarization.THETA_PHI,
        sweaver.Polarization.LUDWIG3_X,
        sweaver.Polarization.LUDWIG3_Y,
    ],
)
def test_evaluate_in_frame_ticra_phase_is_conjugation(
    asymmetric_field: sweaver.ElectricField,
    polarization: sweaver.Polarization,
) -> None:
    R_child_to_base = rotation_matrix_z(np.radians(20.0)) @ rotation_matrix_y(
        np.radians(12.0)
    )

    theta = np.radians(np.array([5.0, 12.0, 19.0]))
    phi = np.radians(np.array([15.0, 90.0, 210.0]))

    e1, e2 = asymmetric_field.evaluate_in_frame(
        theta_rad=theta,
        phi_rad=phi,
        rotation_matrix_child_to_base=R_child_to_base,
        polarization=polarization,
        epsilon=1e-9,
        use_ticra_phase=False,
    )

    e1_ticra, e2_ticra = asymmetric_field.evaluate_in_frame(
        theta_rad=theta,
        phi_rad=phi,
        rotation_matrix_child_to_base=R_child_to_base,
        polarization=polarization,
        epsilon=1e-9,
        use_ticra_phase=True,
    )

    npt.assert_allclose(e1_ticra, np.conj(e1), rtol=0.0, atol=1e-13)
    npt.assert_allclose(e2_ticra, np.conj(e2), rtol=0.0, atol=1e-13)


@pytest.mark.parametrize(
    "data_file,expected_header,polarization",
    [
        (
            "asymmetric_grid_thetaphi.grd.gz",
            (1, 1, 2, 7),
            sweaver.Polarization.THETA_PHI,
        ),
        (
            "asymmetric_grid_ludwig.grd.gz",
            (1, 3, 2, 7),
            sweaver.Polarization.LUDWIG3_X,
        ),
    ],
)
def test_evaluate_at_locs_matches_ticra_grid(
    data_file: str,
    expected_header: tuple[int, int, int, int],
    polarization: sweaver.Polarization,
) -> None:
    electric_field = load_asymmetric_field()

    with gzip.open(sweaver.get_test_data_path(data_file), "rt") as f:
        grid = load_grd_file(f, expected_header=expected_header)

    theta = np.linspace(grid.theta_start_rad, grid.theta_end_rad, grid.ntheta)
    phi = np.linspace(grid.phi_start_rad, grid.phi_end_rad, grid.nphi)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")

    e1, e2 = electric_field.evaluate_at_locs(
        theta_rad=theta_grid,
        phi_rad=phi_grid,
        polarization=polarization,
        epsilon=1e-9,
    )

    npt.assert_allclose(e1.flatten(), grid.e_field[:, 0], atol=1e-4)
    npt.assert_allclose(e2.flatten(), grid.e_field[:, 1], atol=1e-4)


def test_evaluate_cut_in_frame_identity_matches_evaluate_cut(
    asymmetric_field: sweaver.ElectricField,
) -> None:
    phi = np.radians(37.0)
    theta_start = np.radians(-12.0)
    theta_end = np.radians(18.0)
    ntheta = 31

    e1_frame, e2_frame = asymmetric_field.evaluate_cut_in_frame(
        phi_angle_rad=phi,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        frame=np.eye(3),
        polarization=sweaver.Polarization.LUDWIG3_X,
        epsilon=1e-9,
    )

    e1_cut, e2_cut = asymmetric_field.evaluate_cut(
        phi_angle_rad=phi,
        theta_start_rad=theta_start,
        theta_end_rad=theta_end,
        ntheta=ntheta,
        polarization=sweaver.Polarization.LUDWIG3_X,
        epsilon=1e-9,
    )

    npt.assert_allclose(e1_frame, e1_cut, rtol=1e-10, atol=1e-12)
    npt.assert_allclose(e2_frame, e2_cut, rtol=1e-10, atol=1e-12)


def test_evaluate_theta_phi_grid_in_frame_identity_matches_grid(
    asymmetric_field: sweaver.ElectricField,
) -> None:
    kwargs = dict(
        theta_start_rad=np.radians(2.0),
        theta_end_rad=np.radians(11.0),
        ntheta=8,
        phi_start_rad=np.radians(15.0),
        phi_end_rad=np.radians(72.0),
        nphi=7,
        polarization=sweaver.Polarization.THETA_PHI,
        epsilon=1e-9,
    )

    e1_frame, e2_frame = asymmetric_field.evaluate_theta_phi_grid_in_frame(
        **kwargs,  # ty:ignore[invalid-argument-type]
        frame=np.eye(3),
    )

    e1_grid, e2_grid = asymmetric_field.evaluate_theta_phi_grid(**kwargs)  # ty:ignore[invalid-argument-type]

    npt.assert_allclose(e1_frame, e1_grid, rtol=1e-10, atol=1e-12)
    npt.assert_allclose(e2_frame, e2_grid, rtol=1e-10, atol=1e-12)
