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

import sweaver

import numpy as np
import numpy.testing as npt
import pytest


def get_beam_alm(file_name: str) -> sweaver.Beam:
    with (sweaver.get_test_data_path(file_name)).open("rt") as f:
        grasp_file = sweaver.read_sph_file(f)

    electric_field = sweaver.ElectricField.from_frequency_block(grasp_file.get(index=0))
    return sweaver.Beam.from_electric_field(electric_field, lmax=2)


def test_convert_electric_dipole_to_beam(data_dir):
    beam = get_beam_alm("hertzian_e_dipole_x.sph")
    assert beam.lmax == 2
    assert beam.mmax == 2

    idx_00 = beam.get_idx(0, 0)
    idx_20 = beam.get_idx(2, 0)
    idx_22 = beam.get_idx(2, 2)

    a_00_i = beam.alm_i[idx_00]
    a_20_i = beam.alm_i[idx_20]
    a_22_i = beam.alm_i[idx_22]
    a_20_e = beam.alm_e[idx_20]
    a_22_e = beam.alm_e[idx_22]

    (
        npt.assert_allclose(beam.alm_b, 0 + 0j, atol=1e-15),
        "No mode B should be present in a dipole",
    )

    assert np.isclose(a_00_i.imag, 0, atol=1e-15), "The monopole must be real"
    assert a_00_i.real > 0, "The monopole must be positive"

    assert np.isclose(a_20_i.imag, 0, atol=1e-15), "The m=0 quadrupole must be real"
    assert a_20_i.real > 0, "The quadrupole must be positive"

    assert np.isclose(a_22_i.imag, 0, atol=1e-15), "The m=2 quadrupole must be real"
    assert a_22_i.real < 0, "The m=2 quadrupole must be negative"

    # Since the field is real, a_{l,-m} = (-1)^m a_{l,m}^*
    # For m=2, (-1)^2 = 1. So they should be equal.
    # We verify the ratio of power in m=0 vs m=2 roughly fits
    # (Exact ratio depends on normalization, but orders of magnitude should match)
    assert 0.5 < abs(a_22_i / a_20_i) < 2.0, "Inconsistent quadrupole for m = ±1"

    assert np.isclose(a_20_e.imag, 0, atol=1e-15)
    assert a_20_e.real > 0

    assert np.isclose(a_22_e.imag, 0, atol=1e-15)
    assert a_22_e.real < 0


def test_beam_get_alms_behavior(data_dir):
    beam = get_beam_alm("hertzian_e_dipole_x.sph")

    # m = 0
    idx_20 = beam.get_idx(2, 0)
    i_20, e_20, b_20 = beam.get_alms(2, 0)
    assert i_20 == beam.alm_i[idx_20]
    assert e_20 == beam.alm_e[idx_20]
    assert b_20 == beam.alm_b[idx_20]

    # m > 0
    idx_21 = beam.get_idx(2, 1)
    i_21, e_21, b_21 = beam.get_alms(2, 1)
    assert i_21 == beam.alm_i[idx_21]
    assert e_21 == beam.alm_e[idx_21]

    # m < 0 with |m| even
    # a_{l, -2} = (-1)^(-2) * a_{l, 2}^* = + a_{l, 2}^*
    idx_22 = beam.get_idx(2, 2)
    i_m2, e_m2, b_m2 = beam.get_alms(2, -2)
    assert i_m2 == pytest.approx(np.conj(beam.alm_i[idx_22]), abs=1e-15)
    assert e_m2 == pytest.approx(np.conj(beam.alm_e[idx_22]), abs=1e-15)
    assert b_m2 == pytest.approx(np.conj(beam.alm_b[idx_22]), abs=1e-15)

    # m < 0 with |m| odd
    # a_{l, -1} = (-1)^(-1) * a_{l, 1}^* = - a_{l, 1}^*
    i_m1, e_m1, b_m1 = beam.get_alms(2, -1)
    assert i_m1 == pytest.approx(-np.conj(beam.alm_i[idx_21]), abs=1e-15)
    assert e_m1 == pytest.approx(-np.conj(beam.alm_e[idx_21]), abs=1e-15)
    assert b_m1 == pytest.approx(-np.conj(beam.alm_b[idx_21]), abs=1e-15)

    # Out-of-bounds condition
    with pytest.raises(ValueError, match="out-of-bounds"):
        beam.get_alms(beam.lmax + 1, 0)  # ℓ troppo grande

    with pytest.raises(ValueError, match="out-of-bounds"):
        beam.get_alms(-1, 0)  # ℓ negativo

    # |m| > ℓ or |m| > mmax ---
    # Asking m=2 for ℓ=1 should return 0
    i_out, e_out, b_out = beam.get_alms(1, 2)
    assert i_out == 0j
    assert e_out == 0j
    assert b_out == 0j

    # Here we ask for a m larger than the biggest in the file
    i_out2, e_out2, b_out2 = beam.get_alms(beam.lmax, beam.mmax + 1)
    assert i_out2 == 0j
    assert e_out2 == 0j
    assert b_out2 == 0j


def test_convert_magnetic_dipole_to_beam(data_dir):
    # We load the electric dipole for comparison, as we assume that it was already
    # tested (see `test_convert_electric_dipole_to_beam` above) and we can verify
    # that the expected symmetries between the spin-2 quantities of the magnetic
    # and electric dipoles hold.
    e_beam_reference = get_beam_alm("hertzian_e_dipole_x.sph")

    # This is the variable under test
    beam = get_beam_alm("hertzian_h_dipole_x.sph")

    idx_00 = beam.get_idx(0, 0)
    idx_20 = beam.get_idx(2, 0)
    idx_22 = beam.get_idx(2, 2)

    # Magnetic (The Subject)
    m_a00_i = beam.alm_i[idx_00]
    m_a20_i = beam.alm_i[idx_20]
    m_a20_e = beam.alm_e[idx_20]
    m_a22_e = beam.alm_e[idx_22]
    m_a20_b = beam.alm_b[idx_20]

    # Electric (The Reference)
    e_a00_i = e_beam_reference.alm_i[idx_00]
    e_a20_e = e_beam_reference.alm_e[idx_20]
    e_a22_e = e_beam_reference.alm_e[idx_22]

    ################################################################################
    # CONSERVATION OF ENERGY (INTENSITY)
    # The radiation pattern I(theta, phi) is invariant under E <-> H duality.

    # Monopole Check (Total Power)
    assert np.isclose(m_a00_i.imag, 0, atol=1e-15), (
        "Intensity monopole must be real-valued."
    )

    assert m_a00_i.real > 0, "Total radiated power must be positive."

    assert np.isclose(m_a00_i.real, e_a00_i.real, atol=1e-15), (
        "Violation of Duality: Magnetic dipole must radiate same total power as Electric dipole."
    )

    # Quadrupole Check (Beam Shape)
    assert np.isclose(m_a20_i.imag, 0, atol=1e-15), (
        "Intensity quadrupole must be real-valued."
    )

    assert np.isclose(m_a20_i.real, 0.7926, rtol=1e-3), (
        "Intensity anisotropy (doughnut shape) mismatch for m=0."
    )

    ################################################################################
    # THE "NULL" TEST (MODE SELECTION)
    # A magnetic dipole radiates TE modes. In the far-field limit, this
    # produces linear polarization (E-modes). It does NOT produce Cosmological
    # B-modes (curl component of the polarization field).

    assert np.allclose(beam.alm_b, 0 + 0j, atol=1e-15), (
        "Magnetic dipole (TE mode) should not produce Cosmological B-modes."
    )

    assert np.abs(m_a20_b) < 1e-15, "Specific check: m=0 B-mode component must be null."

    ################################################################################
    # POLARIZATION DUALITY (THE 90° FLIP)
    # The polarization vector of H-dipole is rotated 90° relative to E-dipole.
    # Rot(90) -> (Q, U) -> (-Q, -U).
    # Therefore, all E-mode coefficients must flip their sign but conserve
    # their magnitude

    # The m=0 quadrupole (main lobe polarization)
    assert np.isclose(m_a20_e.imag, 0.0, atol=1e-15), (
        "Polarization E-mode must be real-valued."
    )

    assert m_a20_e.real < 0, (
        "Sign Flip Violation: Magnetic dipole m=0 E-mode must be NEGATIVE (Electric was Positive)."
    )

    assert np.isclose(m_a20_e.real, -e_a20_e.real, rtol=1e-5), (
        "Magnitude Violation: m=0 E-mode magnitude not conserved under duality."
    )

    # The m=2 quadrupole (azimuthal polarization)
    assert np.isclose(m_a22_e.imag, 0, atol=1e-15), (
        "Polarization m=2 E-mode must be real-valued."
    )

    assert m_a22_e.real > 0, (
        "Sign Flip Violation: Magnetic dipole m=2 E-mode must be POSITIVE (Electric was Negative)."
    )

    assert np.isclose(m_a22_e.real, -e_a22_e.real, rtol=1e-5), (
        "Magnitude Violation: m=2 E-mode magnitude not conserved under duality."
    )

    ################################################################################
    # GEOMETRIC CONSISTENCY (X-AXIS ALIGNMENT)
    # The ratio of m=2 to m=0 modes defines the orientation of the dipole.
    # This ratio must be consistent regardless of E or H type.

    ratio = abs(m_a22_e / m_a20_e)
    assert 0.5 < ratio < 2.0, (
        f"Geometric Distortion: Quadrupole shape ratio {ratio:.2f} is inconsistent with x-axis alignment."
    )


def test_angular_power_spectra_math(data_dir):
    # We load the file containing the dipole, but it’s just to have
    # the arrays already allocated with lmax=2
    beam = get_beam_alm("hertzian_e_dipole_x.sph")

    # Clean up everything in the arrays of a_ℓm coefficients
    beam.alm_i.fill(0j)
    beam.alm_e.fill(0j)
    beam.alm_b.fill(0j)

    # Scegliamo di testare il multipolo ℓ = 2.
    # La formula dello spettro per un campo reale è:
    # C_2 = ( |a_{2,0}|^2 + 2 * |a_{2,1}|^2 + 2 * |a_{2,2}|^2 ) / (2*2 + 1)

    # Simple real values for the I coefficients
    beam.alm_i[beam.get_idx(2, 0)] = 3.0 + 0j
    beam.alm_i[beam.get_idx(2, 1)] = 2.0 + 0j
    beam.alm_i[beam.get_idx(2, 2)] = 1.0 + 0j

    # For the E coefficients, we put the same values
    beam.alm_e[beam.get_idx(2, 0)] = 1.0 + 0j
    beam.alm_e[beam.get_idx(2, 1)] = 1.0 + 0j
    beam.alm_e[beam.get_idx(2, 2)] = 1.0 + 0j

    # Compute the spectra
    ells, cl_i, cl_e, cl_b = beam.angular_power_spectra(ell_start=2)

    assert len(ells) == 1
    assert ells[0] == 2

    # I: ( 3² + 2*(2²) + 2*(1²) ) / 5 = (9 + 8 + 2) / 5 = 19 / 5 = 3.8
    expected_cl_i = 3.8
    assert cl_i[0] == pytest.approx(expected_cl_i, abs=1e-12)

    # E: ( 1² + 2*(1²) + 2*(1²) ) / 5 = (1 + 2 + 2) / 5 = 5 / 5 = 1.0
    expected_cl_e = 1.0
    assert cl_e[0] == pytest.approx(expected_cl_e, abs=1e-12)

    # B was zero
    assert cl_b[0] == pytest.approx(0.0, abs=1e-12)
