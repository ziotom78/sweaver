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
import sweaver


def build_test_field(
    lmax: int, mmax: int, active_modes: list[tuple[int, int, int, complex]]
) -> sweaver.ElectricField:
    """
    Helper function to create a mathematically pure ElectricField.
    Injects specific values into the alm_stack at the given (component, ℓ, m) coordinates.
    """
    # Calculate exact memory size for the HEALPix/ducc0 layout
    nalm = (mmax + 1) * (2 * lmax + 2 - mmax) // 2
    alm_stack = np.zeros((4, nalm), dtype=np.complex128)

    for comp, ell, m, val in active_modes:
        idx = sweaver.ElectricField._get_idx(ell, m, lmax=lmax)
        alm_stack[comp, idx] = val

    return sweaver.ElectricField(
        frequency_ghz=30.0, lmax=lmax, mmax=mmax, alm_stack=alm_stack
    )


def test_total_power_null():
    """A field with zero amplitude everywhere must contain exactly zero power."""
    field = build_test_field(lmax=5, mmax=5, active_modes=[])

    assert field.total_power() == 0.0


def test_total_power_m0_only():
    """
    Modes with m=0 represent azimuthally symmetric (zonal) patterns.
    Because there is no m=-0, the Parseval weight is exactly 1.0.
    """
    # Inject amplitude 5.0 into the Real E-mode (comp 0) at l=1, m=0
    # Expected power: |5.0|^2 * 1 = 25.0
    field = build_test_field(lmax=5, mmax=5, active_modes=[(0, 1, 0, 5.0 + 0.0j)])

    np.testing.assert_allclose(field.total_power(), 25.0, atol=1e-12)


def test_total_power_m_positive():
    """
    Modes with m > 0 implicitly store the power for m < 0 due to the real-map
    SHT symmetry in ducc0. The Parseval weight must be exactly 2.0.
    """
    # Inject amplitude 3.0 into the Real B-mode (comp 1) at l=2, m=2
    # Expected power: |3.0|^2 * 2 = 18.0
    field = build_test_field(lmax=5, mmax=5, active_modes=[(1, 2, 2, 3.0 + 0.0j)])

    np.testing.assert_allclose(field.total_power(), 18.0, atol=1e-12)


def test_total_power_complex_mixed():
    """
    Test a mix of m=0, m>0, and complex amplitudes spread across the
    Real and Imaginary electric field components.
    """
    active_modes = [
        # Component 0 (Real E), l=1, m=0
        # Power = |1 + 2j|^2 * 1 = (1^2 + 2^2) * 1 = 5.0
        (0, 1, 0, 1.0 + 2.0j),
        # Component 2 (Imag E), l=3, m=1
        # Power = |0 + 4j|^2 * 2 = (4^2) * 2 = 32.0
        (2, 3, 1, 0.0 + 4.0j),
    ]

    field = build_test_field(lmax=5, mmax=5, active_modes=active_modes)

    # Total expected power = 5.0 + 32.0 = 37.0
    np.testing.assert_allclose(field.total_power(), 37.0, atol=1e-12)
