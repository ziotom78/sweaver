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

import numpy as np
import scipy.constants

import sweaver


def create_analytical_dipole(
    freq_ghz: float, lmax: int, mmax: int
) -> sweaver.ElectricField:
    """
    Creates a mathematically pure, synthetic electric field representing
    a single non-zero harmonic mode (e.g., an l=1, m=1 dipole).
    This provides a clean, analytical baseline for translation tests.
    """
    nalm = ((mmax + 1) * (mmax + 2)) // 2 + (mmax + 1) * (lmax - mmax)
    alm_stack = np.zeros((4, nalm), dtype=np.complex128)

    # Get the ducc0 index for l=1, m=1
    idx = sweaver.ElectricField._get_idx(lmax=lmax, ell=1, m=1)

    # Inject 1.0 + 0j into the E-mode (Real part) of the l=1, m=1 harmonic
    alm_stack[0, idx] = 1.0 + 0.0j

    return sweaver.ElectricField(freq_ghz, lmax, mmax, alm_stack)


def test_translate_zero():
    """A translation of (0, 0, 0) should perfectly recover the original field."""
    original = create_analytical_dipole(freq_ghz=30.0, lmax=10, mmax=10)

    translated = original.translate_phase_center(dx_m=0.0, dy_m=0.0, dz_m=0.0)

    # The lmax/mmax should not have grown
    assert translated.lmax == original.lmax
    assert translated.mmax == original.mmax

    # The coefficients must perfectly match (within numerical precision)
    np.testing.assert_allclose(translated.alm_stack, original.alm_stack, atol=1e-12)


def test_translate_z_preserves_symmetry():
    """
    Translating a beam strictly along the Z-axis preserves azimuthal symmetry.
    Therefore, if the original beam only has power in m=1, the translated
    beam must ALSO only have power in m=1.
    """
    original = create_analytical_dipole(freq_ghz=30.0, lmax=10, mmax=10)

    # Translate by 10 wavelengths along Z
    dz = 10 * (scipy.constants.speed_of_light / 30e9)
    translated = original.translate_phase_center(dx_m=0.0, dy_m=0.0, dz_m=dz)

    # Verify that power leaked into higher ℓ modes (the beam got wider).
    # Due to parity conservation, multiplying an ℓ=1 (odd) dipole by the real (even)
    # part of the Z-axis phase shift yields strictly odd ℓ-modes (ℓ=3, 5, 7…).
    idx_l3_m1 = sweaver.ElectricField._get_idx(lmax=translated.lmax, ell=3, m=1)
    assert np.abs(translated.alm_stack[0, idx_l3_m1]) > 1e-10, (
        "Z-shift must excite ℓ=3 in the Real part"
    )

    # Conversely, multiplying by the imaginary (odd) part of the phase shift
    # yields strictly even l-modes (ℓ=2, 4, 6...).
    idx_l2_m1 = sweaver.ElectricField._get_idx(lmax=translated.lmax, ell=2, m=1)
    assert np.abs(translated.alm_stack[2, idx_l2_m1]) > 1e-10, (
        "Z-shift must excite ℓ=2 in the Imaginary part"
    )

    # Verify that ALL modes where m ≠ 1 are strictly zero
    for ell in range(translated.lmax + 1):
        for m in range(min(ell, translated.mmax) + 1):
            if m != 1:
                idx = sweaver.ElectricField._get_idx(lmax=translated.lmax, ell=ell, m=m)
                np.testing.assert_allclose(
                    translated.alm_stack[:, idx],
                    0.0,
                    atol=1e-12,
                    err_msg=f"Mode ℓ={ell}, m={m} should be zero for Z-axis translation.",
                )


def test_translate_x_breaks_symmetry():
    """
    Translating a beam along the X-axis breaks its azimuthal symmetry relative
    to the global origin, forcing power into adjacent m-modes.
    """
    original = create_analytical_dipole(freq_ghz=30.0, lmax=10, mmax=10)

    # Translate by 10 wavelengths along X
    dx = 10 * (scipy.constants.speed_of_light / 30e9)
    translated = original.translate_phase_center(dx_m=dx, dy_m=0.0, dz_m=0.0)

    # Verify that m=0 and m=2 are now populated (symmetry broken).
    # Because the X-translation phase factor introduces an imaginary cross-term
    # (via the Jacobi-Anger expansion), the newly excited modes will appear
    # in the Imaginary part of the electric field.
    # To make the test completely robust against phase shifts, we check
    # the total magnitude across all 4 (Real/Imag, E/B) components.
    idx_m0 = sweaver.ElectricField._get_idx(lmax=translated.lmax, ell=2, m=0)
    idx_m2 = sweaver.ElectricField._get_idx(lmax=translated.lmax, ell=2, m=2)

    power_m0 = np.sum(np.abs(translated.alm_stack[:, idx_m0]))
    power_m2 = np.sum(np.abs(translated.alm_stack[:, idx_m2]))

    assert power_m0 > 1e-10, "X-shift must excite m=0 modes"
    assert power_m2 > 1e-10, "X-shift must excite m=2 modes"


def test_translate_round_trip():
    """
    A beam translated by +d and then back by -d must exactly recover its original
    state, demonstrating that the SHT upsampling logic is lossless and alias-free.
    """
    original = create_analytical_dipole(freq_ghz=30.0, lmax=10, mmax=10)

    dx, dy, dz = 0.1, -0.05, 0.2  # Arbitrary 3D shift

    # Forward translation (upsamples to larger lmax)
    shifted_forward = original.translate_phase_center(dx_m=dx, dy_m=dy, dz_m=dz)

    # Backward translation (back to origin)
    shifted_back = shifted_forward.translate_phase_center(dx_m=-dx, dy_m=-dy, dz_m=-dz)

    # Check that the original low-ℓ modes are perfectly preserved
    # We loop over the original, smaller array size to compare
    for ell in range(original.lmax + 1):
        for m in range(min(ell, original.mmax) + 1):
            idx_orig = sweaver.ElectricField._get_idx(lmax=original.lmax, ell=ell, m=m)
            idx_back = sweaver.ElectricField._get_idx(
                lmax=shifted_back.lmax, ell=ell, m=m
            )

            np.testing.assert_allclose(
                shifted_back.alm_stack[:, idx_back],
                original.alm_stack[:, idx_orig],
                atol=1e-12,
                err_msg=f"Round-trip failed to conserve power at ℓ={ell}, m={m}",
            )
