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


def create_dummy_field(
    freq_ghz: float, lmax: int, mmax: int, base_multiplier: float
) -> sweaver.ElectricField:
    """
    Creates a dummy ElectricField where every (ℓ, m) coefficient is uniquely
    fingerprinted using the function: base_multiplier * (10 * (ℓ + 1)**2 + m)
    """
    nalm = mmax * (2 * lmax + 1 - mmax) // 2 + lmax + 1
    alm_stack = np.zeros((4, nalm), dtype=np.complex128)

    for ell in range(lmax + 1):
        for m in range(min(ell, mmax) + 1):
            idx = sweaver.ElectricField._get_idx(ell=ell, m=m, lmax=lmax)

            # The unique fingerprint
            val = base_multiplier * (10 * (ell + 1) ** 2 + m)
            alm_stack[:, idx] = val + 0.0j

    return sweaver.ElectricField(freq_ghz, lmax, mmax, alm_stack)


def test_add_cross_sizes():
    """
    Test that summing fields works even if one has a larger `lmax` and
    another has a larger `mmax`.
    """
    # Field A values will be in the tens/hundreds (e.g., 40.0)
    field_a = create_dummy_field(30.0, lmax=15, mmax=2, base_multiplier=1.0)

    # Field B values will be in the thousands (e.g., 40000.0)
    field_b = create_dummy_field(30.0, lmax=5, mmax=5, base_multiplier=1000.0)

    field_c = field_a + field_b

    assert field_c.lmax == 15
    assert field_c.mmax == 5

    for ell in range(field_c.lmax + 1):
        for m in range(min(ell, field_c.mmax) + 1):
            idx = sweaver.ElectricField._get_idx(lmax=field_c.lmax, ell=ell, m=m)

            in_a = (ell <= field_a.lmax) and (m <= field_a.mmax)
            in_b = (ell <= field_b.lmax) and (m <= field_b.mmax)

            base_val = 10 * (ell + 1) ** 2 + m
            expected = 0.0 + 0.0j

            if in_a:
                expected += 1.0 * base_val
            if in_b:
                expected += 1000.0 * base_val

            np.testing.assert_allclose(
                field_c.alm_stack[:, idx],
                expected,
                atol=1e-12,
                err_msg=f"Addition alignment failed at ℓ={ell}, m={m}",
            )


def test_sub_cross_sizes():
    """
    Test subtraction of electric fields.
    """
    field_a = create_dummy_field(30.0, lmax=10, mmax=2, base_multiplier=1.0)
    field_b = create_dummy_field(30.0, lmax=5, mmax=5, base_multiplier=1000.0)

    field_c = field_a - field_b

    assert field_c.lmax == 10
    assert field_c.mmax == 5

    for ell in range(field_c.lmax + 1):
        for m in range(min(ell, field_c.mmax) + 1):
            idx = sweaver.ElectricField._get_idx(lmax=field_c.lmax, ell=ell, m=m)

            in_a = (ell <= field_a.lmax) and (m <= field_a.mmax)
            in_b = (ell <= field_b.lmax) and (m <= field_b.mmax)

            base_val = 10 * (ell + 1) ** 2 + m
            expected = 0.0 + 0.0j

            if in_a:
                expected += 1.0 * base_val
            if in_b:
                expected -= 1000.0 * base_val

            np.testing.assert_allclose(
                field_c.alm_stack[:, idx],
                expected,
                atol=1e-12,
                err_msg=f"Subtraction alignment failed at ℓ={ell}, m={m}",
            )
