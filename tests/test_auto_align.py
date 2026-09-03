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
import pytest

import sweaver

from utils import get_gaussian_beam, angle_mod_pi_distance, angular_distance_periodic


def test_find_peak():
    """Verifies that find_peak accurately reports the injected coordinates."""

    gaussian_efield = get_gaussian_beam()

    theta_off = np.radians(5.0)
    phi_off = np.radians(30.0)

    # Inject the displacement (direct rotation)
    displaced_efield = gaussian_efield.rotate_euler(
        sweaver.EulerAngles(alpha_rad=phi_off, beta_rad=theta_off, gamma_rad=0.0),
    )

    # Run the locator
    th, ph, ps = displaced_efield.find_peak(
        region_theta_rad=(0, 2 * theta_off, 40), region_phi_rad=(0, 2 * np.pi, 80)
    )

    assert th == pytest.approx(theta_off, abs=1e-3)
    assert ph == pytest.approx(phi_off, abs=1e-3)


def test_get_alignment_angles_for_simple_displacement():
    gaussian_efield = get_gaussian_beam()

    theta_off = np.radians(8.0)
    phi_off = np.radians(120.0)

    displaced_efield = gaussian_efield.rotate_euler(
        sweaver.EulerAngles(
            alpha_rad=phi_off,
            beta_rad=theta_off,
            gamma_rad=0.0,
        )
    )

    correction_angles = displaced_efield.get_alignment_angles(
        region_theta_rad=(0, 2 * theta_off, 40),
        region_phi_rad=(0, 2 * np.pi, 80),
    )

    # The polarization twist is only defined modulo pi for this linearly
    # polarized Gaussian test case.
    assert angle_mod_pi_distance(
        correction_angles.alpha_rad,
        0.0,
    ) == pytest.approx(0.0, abs=1e-3)

    assert correction_angles.beta_rad == pytest.approx(-theta_off, abs=1e-3)

    assert angular_distance_periodic(
        correction_angles.gamma_rad,
        -phi_off,
    ) == pytest.approx(0.0, abs=1e-3)
